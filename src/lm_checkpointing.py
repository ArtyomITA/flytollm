"""Optional activation recomputation; identical LM equations and functional state."""
import torch
from torch.utils.checkpoint import checkpoint
from fly_lm import LMState,state_tensors
from fly_attention import AttentionCache


def unpack(values):
    return LMState(values[0],values[1],AttentionCache(*values[2:]))


def checkpoint_forward(model,ids,state=None,active=None,reset=None,segment=2):
    if segment<1:raise ValueError('positive segment length required')
    if state is None:state=model.initial_state(ids.shape[1])
    if active is None:active=ids.ne(model.config.pad)
    if reset is None:reset=torch.zeros_like(ids,dtype=torch.bool)
    def run(tokens,enabled,clear,*flat_state):
        logits,new=model(tokens,unpack(flat_state),enabled,clear)
        return (logits,*state_tensors(new))
    outputs=[]
    for start in range(0,len(ids),segment):
        values=checkpoint(run,ids[start:start+segment],active[start:start+segment],reset[start:start+segment],
                          *state_tensors(state),use_reentrant=False)
        outputs.append(values[0]);state=unpack(values[1:])
    return torch.cat(outputs),state
