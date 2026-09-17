"""Integrity/roundtrip checks; no LM scores, no heldout text printed."""
import json
import sqlite3
import numpy as np
from tokenizers import Tokenizer
from prepare_text import OUT, SOURCES, stories, digest, sha, write_json, emit
from text_dataset import StoryDataset


def main():
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf8'))
    audit=json.loads((OUT/'audit.json').read_text(encoding='utf8'))
    upstream=json.loads((OUT/'upstream_metadata.json').read_text(encoding='utf8'))
    source_oids={x['path']:x['lfs']['oid'] for x in upstream['files']}
    tok=Tokenizer.from_file(str(OUT/'tokenizer-4096.json'))
    assert sha(OUT/'tokenizer-4096.json') == manifest['tokenizer_sha256']
    assert sha(OUT/'audit.json') == manifest['audit_sha256']
    for split,path in SOURCES.items():
        assert sha(path) == audit['sources'][split]['source_sha256'] == source_oids[path.name]
    results={}
    for split,info in manifest['splits'].items():
        for ext,meta in info['files'].items():
            assert sha(OUT/f'{split}.{ext}') == meta['sha256']
        with StoryDataset(OUT,split) as ds:
            assert np.all(ds.offsets[1:]>ds.offsets[:-1])
            assert np.all(ds.tokens[ds.offsets[:-1].astype(np.int64)] == 1)
            assert np.all(ds.tokens[ds.offsets[1:].astype(np.int64)-1] == 2)
            special_counts=np.zeros(3,dtype=np.int64)
            for start in range(0,len(ds.tokens),2**20):
                block=ds.tokens[start:start+2**20]
                assert block.max() < 4096
                for special in range(3):
                    special_counts[special] += np.count_nonzero(block==special)
            assert special_counts.tolist()==[0,len(ds),len(ds)]
            assert int(ds.byte_counts.sum(dtype=np.uint64))==info['text_bytes']
            results[split]=dict(stories=len(ds),tokens=len(ds.tokens),hashes_ok=True,boundaries_ok=True)
        emit('verified',split=split,**results[split])
    # Source -> serialized tokens -> original retained payload, predetermined first32 per split.
    roundtrips={s:0 for s in results}
    for source in ('valid','train'):
        selection=(OUT/f'{source}.selection.u8').read_bytes()
        datasets={s:StoryDataset(OUT,s) for s in (('validation','test') if source=='valid' else ('train',))}
        counters={s:0 for s in datasets}
        try:
            for i,(text,_) in enumerate(stories(SOURCES[source])):
                code=selection[i]
                if not code: continue
                split={1:'train',2:'validation',3:'test'}[code]
                if roundtrips[split]<32:
                    ids=datasets[split][counters[split]][1:-1].tolist()
                    assert tok.decode(ids,skip_special_tokens=False)==text
                    roundtrips[split]+=1
                counters[split]+=1
                if all(roundtrips[s]>=32 for s in datasets): break
        finally:
            for ds in datasets.values(): ds.close()
    # Freeze evaluation indices by smallest normalized content hashes, without selecting by quality.
    db=sqlite3.connect(f'file:{(OUT/"dedup.sqlite").as_posix()}?mode=ro',uri=True)
    wanted={s:{h:i for i,(h,) in enumerate(db.execute('SELECT hash FROM seen WHERE split=? ORDER BY hash LIMIT ?',
              (s,limit)))} for s,limit in [('test',512),('validation',128)]}
    db.close()
    selected={s:[] for s in wanted}; counters={s:0 for s in wanted}
    selection=(OUT/'valid.selection.u8').read_bytes()
    for i,(text,_) in enumerate(stories(SOURCES['valid'])):
        code=selection[i]
        if code not in (2,3): continue
        split={2:'validation',3:'test'}[code]; h=digest(text)
        if h in wanted[split]:
            selected[split].append(dict(rank=wanted[split][h],story_index=counters[split],sha256=h.hex()))
        counters[split]+=1
    for split in selected:
        selected[split].sort(key=lambda x:x['rank'])
        assert len(selected[split])==len(wanted[split])
    write_json(OUT/'evaluation_indices.json',dict(rule='smallest normalized SHA256; generation first64 test entries',splits=selected))
    write_json(OUT/'verification.json',dict(ok=True,upstream_sha256_match=True,roundtrips=roundtrips,
        splits=results,test_scores_computed=False,evaluation_indices_sha256=sha(OUT/'evaluation_indices.json')))
    emit('verification_complete',roundtrips=roundtrips)


if __name__=='__main__': main()
