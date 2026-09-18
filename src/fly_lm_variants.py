"""Control-code variants of the language-model wrapper (phase 8, section N of SUITE_TEST_FASE_6B.md). fly_lm.FlyLM is
untouched; LoopedFlyLM overrides step() so that the external attention can be read more than once per token, inside the
loop of core substeps, with the same weights at every read (the looped-transformer logic, RICERCA_LOOPED_A/B.md).

reads            substeps (1-based, 1..T-1 with T = pre_steps + post_steps) after which the attention is read; every read
                 feeds back into the graph through the feedback ports (no text -> logits bypass); () = attention never
                 read (its parameters and the feedback injector become buffers: no gradient, no optimizer state).
kv               'final'    one KV cache, keys/values from the final state of each token (the main model's behaviour)
                 'per_read' one KV cache per read: token t at read r attends to the read-r states of tokens < t
                            (Universal Transformer / Huginn / Ouro); caches are stacked along the batch axis, so the
                            state is still one AttentionCache of five tensors (checkpoints and CUDA Graph capture unchanged)
                 'first'    one KV cache written from the state at the FIRST read, queried by every read (MoR recursive
                            KV sharing)
step_id          per-read identifier: elementwise gain (init 1) and bias (init 0) on the query input
inject           'concat': the query input is adapter([state ; token embedding]) with adapter init [I | 0] (Huginn)
token_injection  'all' (token current at every substep, the main model) or 'first' (first substep only)
depth_schedule   ((start_update, depth), ...): the number of substeps per half (pre = post = depth) changes DURING training
                 (user idea of 17 September 2026: 12+12, then 8+8, then 4+4, then the shock back to 12+12). The model
                 always runs pre_steps + post_steps = the maximum depth; substeps beyond the current depth are no-ops (state
                 frozen through the core's `active` mask, not counted in the rates), so one CUDA Graph serves every phase.
                 The phase is read from a GPU counter of training forward calls (calls_per_update per optimizer update;
                 evaluation, under no_grad, does not advance it and runs at the current depth). Only with the default
                 single read after the pre block. No time is saved: the point is the dynamics, not the cost.

Rates: a read uses the mean spike rate of the token so far; the final representation uses the mean rate of the substeps
after the first read (after pre_steps when there is no read). With reads=(pre_steps,), kv='final' and no option the step
is the main model's step (checked by phase8_selftest.py)."""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
from fly_attention import AttentionCache
from fly_lm import FlyLM, LMState


@dataclass(frozen=True)
class LoopConfig:
    reads: tuple = (4,)
    kv: str = 'final'
    step_id: bool = False
    inject: str = 'none'
    token_injection: str = 'all'
    depth_schedule: tuple = ()
    calls_per_update: int = 2
    output_scale: float = 1.0


def parameters_to_buffers_(module):
    """Turn every parameter of `module` (recursively) into a buffer: still saved and moved with the model, never trained.
    The trainer (phase3_t45.FairCapture) requires a gradient on every parameter, so unused parameters cannot stay."""
    for sub in module.modules():
        for name, p in list(sub._parameters.items()):
            if p is not None:
                del sub._parameters[name]
                sub.register_buffer(name, p.detach().clone())


class LoopedFlyLM(FlyLM):
    def __init__(self, core, interfaces, attention, config=None, loop=None):
        super().__init__(core, interfaces, attention, config)
        c = self.config
        self.total = c.pre_steps + c.post_steps
        self.loop = l = loop or LoopConfig(reads=(c.pre_steps,))
        reads = tuple(int(r) for r in l.reads)
        if list(reads) != sorted(set(reads)) or any(r < 1 or r >= self.total for r in reads):
            raise ValueError(f'reads must be increasing substeps in 1..{self.total - 1}, got {reads}')
        if l.kv not in ('final', 'per_read', 'first') or l.inject not in ('none', 'concat') or l.token_injection not in ('all', 'first'):
            raise ValueError(f'invalid loop configuration {l}')
        self.reads = reads
        self.after = reads[0] if reads else c.pre_steps          # the final rate averages the substeps after this one
        self.rows = len(reads) if (l.kv == 'per_read' and reads) else 1
        bounds = set(reads) | {self.after, self.total}
        if l.token_injection == 'first':
            bounds.add(1)
        self.bounds = tuple(sorted(bounds))
        dim = attention.config.dim
        if reads and l.step_id:
            self.read_gain = nn.Parameter(torch.ones(len(reads), dim))
            self.read_bias = nn.Parameter(torch.zeros(len(reads), dim))
        if reads and l.inject == 'concat':
            self.adapter = nn.Linear(2 * dim, dim, bias=False)
            with torch.no_grad():
                self.adapter.weight.copy_(torch.cat((torch.eye(dim), torch.zeros(dim, dim)), dim=1))
        self.schedule = tuple((int(u), int(d)) for u, d in l.depth_schedule)
        if self.schedule:
            if reads != (c.pre_steps,) or l.kv != 'final' or l.token_injection != 'all' or c.pre_steps != c.post_steps:
                raise ValueError('depth_schedule needs the default single read, kv=final and pre_steps == post_steps == maximum depth')
            if self.schedule[0][0] != 0 or any(d < 1 or d > c.pre_steps for _, d in self.schedule) or max(d for _, d in self.schedule) != c.pre_steps:
                raise ValueError(f'depth_schedule must start at update 0 with depths in 1..{c.pre_steps} and reach the maximum, got {self.schedule}')
            self.register_buffer('calls', torch.zeros((), dtype=torch.long))
            self.register_buffer('schedule_starts', torch.tensor([u * int(l.calls_per_update) for u, _ in self.schedule], dtype=torch.long))
            self.register_buffer('schedule_depths', torch.tensor([float(d) for _, d in self.schedule]))
            self.register_buffer('substep_index', torch.arange(c.pre_steps, dtype=torch.float32))
        if not reads:
            parameters_to_buffers_(self.attention)
            parameters_to_buffers_(self.interfaces.feedback)
            gate = self.interfaces._parameters.pop('gate')
            self.interfaces.register_buffer('gate', gate.detach().clone())

    def initial_state(self, batch):
        v, s = self.core.initial_state(batch)
        return LMState(v, s, self.attention.empty(batch * self.rows))

    def output_weight(self):
        """Separate output head (variant 'h1', added by add_separate_head) when present, tied embedding otherwise."""
        return self._parameters['head'] if 'head' in self._parameters else self.interfaces.embedding.weight

    def add_separate_head(self):
        self.head = nn.Parameter(self.interfaces.embedding.weight.detach().clone())
        return self

    def freeze_interfaces_(self):
        """E4 (phase 8): every parameter outside the core (embedding, injectors, readout, attention, head, loop extras)
        becomes a buffer; only the core's parameters remain trainable."""
        parameters_to_buffers_(self.interfaces)
        parameters_to_buffers_(self.attention)
        for name in [n for n, p in list(self._parameters.items()) if p is not None]:
            p = self._parameters.pop(name)
            self.register_buffer(name, p.detach().clone())
        return self

    @staticmethod
    def _row(cache, row, batch):
        return AttentionCache(*(x[row * batch:(row + 1) * batch] for x in cache))

    def current_depth(self):
        phase = (self.calls >= self.schedule_starts).sum() - 1
        # index with a 1-element tensor: a 0-dim tensor index is converted to a Python int (device sync), which is
        # not permitted inside CUDA Graph capture (smoke_shock, 18 September 03:05); the result has shape [1]
        return self.schedule_depths[phase.reshape(1)]

    def forward(self, ids, state=None, active=None, reset=None):
        out = super().forward(ids, state, active, reset)
        if self.schedule and torch.is_grad_enabled():
            self.calls.add_(1)   # captured in the training CUDA Graph; the evaluation graph (no_grad) never advances it
        return out

    def _step_scheduled(self, ids, state, active, reset, w):
        depth = self.current_depth()
        gates = self.substep_index < depth                     # bool[max depth]
        cache = self.attention.reset(state.cache, reset)
        current = self.interfaces.input_current(ids)
        vs = (state.voltage, state.spike)
        total = None
        for k in range(self.config.pre_steps):
            vs, rate = self.core.advance(current, 1, vs, weights=w, active=active & gates[k], reset=reset if k == 0 else None)
            total = rate if total is None else total + rate
        provisional = self.interfaces.representation(vs, total / depth)
        recalled = self.attention.read(provisional, cache, active)
        drive = current + self.interfaces.feedback_current(recalled)
        total = None
        for k in range(self.config.post_steps):
            vs, rate = self.core.advance(drive, 1, vs, weights=w, active=active & gates[k])
            total = rate if total is None else total + rate
        final = self.interfaces.representation(vs, total / depth)
        logits = F.linear(final, self.output_weight()) * self.loop.output_scale
        logits = torch.where(active[:, None], logits, torch.zeros_like(logits))
        return logits, LMState(*vs, self.attention.append(final, cache, active))

    def step(self, ids, state=None, active=None, reset=None, weights=None):
        if ids.ndim != 1 or ids.dtype != torch.long:
            raise ValueError('ids must be int64[B]')
        batch = ids.shape[0]
        if state is None:
            state = self.initial_state(batch)
        active = ids.ne(self.config.pad) if active is None else active & ids.ne(self.config.pad)
        reset = ids.eq(self.config.bos) if reset is None else reset | ids.eq(self.config.bos)
        cache = self.attention.reset(state.cache, reset.repeat(self.rows)) if self.reads else state.cache
        w = self.core.weights() if weights is None else weights
        if self.schedule:
            return self._step_scheduled(ids, state, active, reset, w)
        l = self.loop
        current = self.interfaces.input_current(ids)
        silent = torch.zeros_like(current)
        embedded = self.interfaces.input_norm(self.interfaces.embedding(ids)) if (self.reads and l.inject == 'concat') else None
        vs = (state.voltage, state.spike)
        feedback = None
        provisionals = []
        done = 0
        sum_all = None
        sum_after = None
        for bound in self.bounds:
            steps = bound - done
            drive = current if (l.token_injection == 'all' or done == 0) else silent
            if feedback is not None:
                drive = drive + feedback
            vs, rate = self.core.advance(drive, steps, vs, weights=w, active=active, reset=reset if done == 0 else None)
            weighted = rate * steps
            sum_all = weighted if sum_all is None else sum_all + weighted
            if done >= self.after:
                sum_after = weighted if sum_after is None else sum_after + weighted
            done = bound
            if bound in self.reads:
                index = self.reads.index(bound)
                provisional = self.interfaces.representation(vs, sum_all / done)
                query = provisional
                if embedded is not None:
                    query = self.adapter(torch.cat((provisional, embedded), dim=-1))
                if l.step_id:
                    query = query * self.read_gain[index] + self.read_bias[index]
                row = index if l.kv == 'per_read' else 0
                recalled = self.attention.read(query, self._row(cache, row, batch), active)
                feedback = self.interfaces.feedback_current(recalled)
                provisionals.append(provisional)
        final = self.interfaces.representation(vs, sum_after / (self.total - self.after))
        logits = F.linear(final, self.output_weight()) * self.loop.output_scale
        logits = torch.where(active[:, None], logits, torch.zeros_like(logits))
        if self.reads:
            if l.kv == 'final':
                cache = self.attention.append(final, cache, active)
            elif l.kv == 'first':
                cache = self.attention.append(provisionals[0], cache, active)
            else:
                rows = [self.attention.append(p, self._row(cache, r, batch), active) for r, p in enumerate(provisionals)]
                cache = AttentionCache(*(torch.cat(parts, dim=0) for parts in zip(*rows)))
        return logits, LMState(*vs, cache)

    def specification(self):
        spec = super().specification()
        spec['looped'] = dict(asdict(self.loop), reads=list(self.reads), rows=self.rows, depth_schedule=[list(x) for x in self.schedule])
        return spec


def parse_reads(text, pre_steps, post_steps):
    """'default' -> (pre_steps,); 'off' -> (); 'every' -> 1..T-1; otherwise comma-separated substeps."""
    if text in (None, 'default'):
        return (int(pre_steps),)
    if text == 'off':
        return ()
    if text == 'every':
        return tuple(range(1, int(pre_steps) + int(post_steps)))
    return tuple(int(x) for x in str(text).split(',') if x)


def parse_schedule(text):
    """'0:12,500:8,1000:4,1500:12' -> ((0, 12), (500, 8), (1000, 4), (1500, 12))."""
    if not text:
        return ()
    return tuple((int(a), int(b)) for a, b in (item.split(':') for item in str(text).split(',') if item))
