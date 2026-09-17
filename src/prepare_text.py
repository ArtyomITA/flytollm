"""Streaming, reproducible TinyStories preparation. CPU only; never trains an LM."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

os.environ.setdefault('RAYON_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'dataset' / 'prepared_v1'
SOURCES = {s: ROOT / 'dataset' / f'TinyStoriesV2-GPT4-{s}.txt' for s in ('train', 'valid')}
DELIMITER = b'<|endoftext|>'
SPECIAL = ['<|pad|>', '<|bos|>', '<|eos|>']


def emit(phase, **values):
    print(json.dumps(dict(phase=phase, **values)), flush=True)


def guard():
    if os.name == 'nt':
        from bench_runtime import memory
        m = memory()
        if m['available_gib'] < 1.25:
            raise RuntimeError(f'RAM guard: {m}')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    tmp.replace(path)


def stories(path):
    """Yield every delimited record, plus nonempty unterminated tail; UTF-8 strict."""
    pending = b''
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            pieces = (pending + chunk).split(DELIMITER)
            pending = pieces.pop()
            for piece in pieces:
                yield piece.strip(b' \t\r\n').decode('utf8'), True
            if len(pending) > 2**20:
                raise ValueError('Record exceeds 1 MiB: inspect delimiter/source')
        if pending.strip():
            yield pending.strip(b' \t\r\n').decode('utf8'), False


def digest(text):
    # Whitespace-normalized dedup ONLY. Kept training payload otherwise untouched.
    return hashlib.sha256(' '.join(text.split()).encode('utf8')).digest()


def audit():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'dedup.sqlite'
    if target.exists():
        raise FileExistsError('Audit already started: choose a fresh output directory')
    db = sqlite3.connect(target)
    db.execute('PRAGMA cache_size=-262144')
    db.execute('CREATE TABLE seen (hash BLOB PRIMARY KEY, split TEXT) WITHOUT ROWID')
    seen = {}  # SHA256 only, no story payload; bounded by RAM guard below.
    stats = {}
    # Reserve official validation before scanning train. Same normalized story cannot cross splits.
    for split in ('valid', 'train'):
        counts = dict(records=0, kept=0, empty=0, unterminated=0, quarantined=0,
                      duplicates_internal=0, overlap_heldout=0, utf8_bytes_kept=0)
        with open(OUT / f'{split}.selection.u8', 'wb') as sel, \
             open(OUT / 'tokenizer_train.jsonl', 'a', encoding='utf8', newline='\n') as sample, \
             open(OUT / 'tokenizer_probe.jsonl', 'a', encoding='utf8', newline='\n') as probe:
            for i, (text, terminated) in enumerate(stories(SOURCES[split])):
                counts['records'] += 1
                code = 0
                if not text:
                    counts['empty'] += 1
                elif not terminated:
                    counts['unterminated'] += 1
                elif split == 'valid' and i == 0:
                    # Local source begins mid-sentence. Quarantine before any score or tokenizer fit.
                    counts['quarantined'] += 1
                else:
                    h = digest(text)
                    old = seen.get(h)
                    if old:
                        key = 'overlap_heldout' if split == 'train' and old != 'train' else 'duplicates_internal'
                        counts[key] += 1
                    else:
                        # Validation source: 80% development, 20% locked internal test by content hash.
                        dest = ('test' if int.from_bytes(h[:8], 'big') % 5 == 0 else 'validation') if split == 'valid' else 'train'
                        seen[h] = dest
                        code = {'train': 1, 'validation': 2, 'test': 3}[dest]
                        counts['kept'] += 1
                        counts[dest + '_stories'] = counts.get(dest + '_stories', 0) + 1
                        counts['utf8_bytes_kept'] += len(text.encode('utf8'))
                        if split == 'train' and h[0] < 4:
                            sample.write(json.dumps(text, ensure_ascii=False) + '\n')
                        elif split == 'train' and h[0] == 4:
                            probe.write(json.dumps(text, ensure_ascii=False) + '\n')
                sel.write(bytes([code]))
                if i % 10000 == 0:
                    guard()
                if i % 100000 == 0:
                    emit('audit', split=split, **counts)
        counts.update(source=str(SOURCES[split]), source_bytes=SOURCES[split].stat().st_size,
                      source_sha256=sha(SOURCES[split]), selection_sha256=sha(OUT / f'{split}.selection.u8'))
        stats[split] = counts
        emit('audit_done', split=split, **counts)
    emit('audit_index', unique=len(seen))
    db.executemany('INSERT INTO seen VALUES (?,?)', seen.items())
    db.commit()
    db.close()
    write_json(OUT / 'audit.json', dict(schema=1, sources=stats,
        policy='Strict UTF8; strip boundary ASCII whitespace; exclude unterminated tails and valid record0; '
               'dedup SHA256 of Unicode whitespace collapsed; heldout wins over train; '
               'valid hash modulo5==0 test else validation; no near-duplicate semantic audit.',
        sample_sha256=sha(OUT / 'tokenizer_train.jsonl'), probe_sha256=sha(OUT / 'tokenizer_probe.jsonl')))


def jsonlines(path):
    with open(path, encoding='utf8') as f:
        for line in f:
            yield json.loads(line)


def train_tokenizers():
    import numpy as np
    import tokenizers
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    if not (OUT / 'audit.json').exists():
        raise RuntimeError('Complete audit first')
    if (OUT / 'manifest.json').exists():
        raise RuntimeError('Completed dataset immutable: choose a new version for tokenizer changes')
    results = []
    for vocab in (2048, 4096, 8192):
        guard()
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=SPECIAL,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
        begin = time.perf_counter()
        emit('tokenizer_train', vocab=vocab)
        tok.train_from_iterator(jsonlines(OUT / 'tokenizer_train.jsonl'), trainer=trainer)
        path = OUT / f'tokenizer-{vocab}.json'
        tok.save(str(path))
        train_s = time.perf_counter() - begin
        lengths, total_bytes, roundtrips = [], 0, 0
        begin = time.perf_counter()
        for text in jsonlines(OUT / 'tokenizer_probe.jsonl'):
            ids = tok.encode(text, add_special_tokens=False).ids
            if tok.decode(ids, skip_special_tokens=False) != text:
                raise AssertionError('Byte-level round-trip failed')
            lengths.append(len(ids))
            total_bytes += len(text.encode('utf8'))
            roundtrips += 1
        for text in ['caffè 🪰 中文\n\t x', '  leading  trailing  ', '\r\n', '', ''.join(SPECIAL)]:
            assert tok.decode(tok.encode(text).ids, skip_special_tokens=False) == text
        elapsed = time.perf_counter() - begin
        row = dict(vocab=tok.get_vocab_size(), sha256=sha(path), train_seconds=train_s,
            probe_seconds=elapsed, probe_stories=roundtrips, probe_bytes=total_bytes,
            probe_tokens=sum(lengths), bytes_per_token=total_bytes/sum(lengths),
            length_percentiles=dict(zip(['p50','p90','p99','max'], np.percentile(lengths,[50,90,99,100]).tolist())),
            roundtrip_ok=True, embedding_parameters=tok.get_vocab_size()*256)
        results.append(row)
        emit('tokenizer_done', **row)
    write_json(OUT / 'tokenizer_comparison.json', dict(tokenizers_version=tokenizers.__version__,
        selected_vocab=4096, decision='Preserve agreed 4k baseline; offline compression alone does not establish LM quality/speed.',
        candidates=results, special_ids={t: i for i,t in enumerate(SPECIAL)}))


def encode_all():
    import numpy as np
    from tokenizers import Tokenizer
    if (OUT / 'manifest.json').exists():
        raise RuntimeError('Completed dataset immutable: choose a new output version')
    audit_data = json.loads((OUT / 'audit.json').read_text(encoding='utf8'))
    tok = Tokenizer.from_file(str(OUT / 'tokenizer-4096.json'))
    probe = list(jsonlines(OUT / 'tokenizer_probe.jsonl'))[:256]
    expected = tok.encode_batch(probe, add_special_tokens=False)
    fast = tok.encode_batch_fast(probe, add_special_tokens=False)
    assert [e.ids for e in expected] == [e.ids for e in fast]
    handles = {s: open(OUT / f'{s}.tokens.u16', 'wb') for s in ('train', 'validation', 'test')}
    offsets = {s: open(OUT / f'{s}.offsets.u64', 'wb') for s in handles}
    bytes_files = {s: open(OUT / f'{s}.bytes.u32', 'wb') for s in handles}
    totals = {s: dict(stories=0, tokens=0, text_bytes=0) for s in handles}
    for f in offsets.values():
        f.write(np.asarray([0], dtype='<u8').tobytes())

    def flush(batch):
        if not batch:
            return
        encodings = tok.encode_batch_fast([text for dest,text in batch], add_special_tokens=False)
        for (dest,text), enc in zip(batch, encodings):
            # Explicit specials once per story; no hidden post-processor or padding.
            ids = [1, *enc.ids, 2]
            if any(i < 3 for i in enc.ids):
                raise ValueError('Literal reserved special token in source; escape policy required')
            handles[dest].write(np.asarray(ids, dtype='<u2').tobytes())
            t = totals[dest]
            t['stories'] += 1
            t['tokens'] += len(ids)
            t['text_bytes'] += len(text.encode('utf8'))
            offsets[dest].write(np.asarray([t['tokens']], dtype='<u8').tobytes())
            bytes_files[dest].write(np.asarray([len(text.encode('utf8'))], dtype='<u4').tobytes())

    try:
        for source in ('valid', 'train'):
            expected = audit_data['sources'][source]
            if sha(SOURCES[source]) != expected['source_sha256']:
                raise RuntimeError('Source changed after audit')
            selections = (OUT / f'{source}.selection.u8').read_bytes()
            if sha(OUT / f'{source}.selection.u8') != expected['selection_sha256']:
                raise RuntimeError('Selection changed')
            batch = []
            count = 0
            for i, (text, terminated) in enumerate(stories(SOURCES[source])):
                count += 1
                code = selections[i]
                if code:
                    batch.append(({1:'train',2:'validation',3:'test'}[code], text))
                if len(batch) >= 512:
                    flush(batch)
                    batch = []
                    guard()
                if i % 25000 == 0:
                    emit('encode', source=source, records=i, totals=totals)
            flush(batch)
            assert count == len(selections)
    finally:
        for f in [*handles.values(), *offsets.values(), *bytes_files.values()]:
            f.close()
    artifacts = {}
    for split in handles:
        artifacts[split] = dict(**totals[split], files={})
        for ext in ('tokens.u16', 'offsets.u64', 'bytes.u32'):
            p = OUT / f'{split}.{ext}'
            artifacts[split]['files'][ext] = dict(bytes=p.stat().st_size, sha256=sha(p))
        idx = np.memmap(OUT / f'{split}.offsets.u64', dtype='<u8', mode='r')
        assert len(idx) == totals[split]['stories'] + 1 and int(idx[-1]) == totals[split]['tokens']
        assert np.all(np.diff(idx) >= 3)
    write_json(OUT / 'manifest.json', dict(schema=1, complete=True, tokenizer_sha256=sha(OUT/'tokenizer-4096.json'),
        audit_sha256=sha(OUT/'audit.json'), script_sha256=sha(Path(__file__)),
        layout='little-endian u16 tokens; u64 story offsets incl final sentinel; u32 UTF8 payload bytes; BOS1/EOS2, PAD0 absent',
        splits=artifacts, test_policy='Locked internal test; no LM scores or manual prompt selection in phase1'))
    emit('complete', totals=totals)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['audit','tokenizers','encode'])
    args = p.parse_args()
    guard()
    {'audit':audit, 'tokenizers':train_tokenizers, 'encode':encode_all}[args.stage]()
