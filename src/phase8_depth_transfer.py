"""N6c (SUITE_TEST_FASE_6B.md section M): does a model trained at one depth keep working at another depth?
Inference only. A plain-LIF control checkpoint trained with pre = post = d substeps is evaluated on DEV with
pre = post = 4, 8 and 12, same weights, no training. It is the viability check of a depth schedule over training
(user question of 17 September 2026 20:20: 12+12 for the first 2000 updates, then 8+8, then 4+4, then 12+12 again): if the
cross-entropy jumps when the depth changes, every switch of the schedule costs a recovery phase.
Run (GPU idle): .venv/Scripts/python.exe phase8_depth_transfer.py --checkpoint-path results/<run>.latest.pt --output results/phase8_depth_<tag>.json"""
import argparse, json, time
from dataclasses import replace
from pathlib import Path
from phase6c_inference import Suite
from phase8_c15_revert_weights import dev_ce


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-path', required=True)
    p.add_argument('--depths', type=int, nargs='+', default=[4, 8, 12])
    p.add_argument('--output', required=True)
    a = p.parse_args(); started = time.time()
    path = Path(a.checkpoint_path); suite = Suite(path.stem, path); model = suite.model
    trained = (model.config.pre_steps, model.config.post_steps)
    out = dict(checkpoint=str(path), trained_depth=list(trained), ce={})
    for d in a.depths:
        model.config = replace(model.config, pre_steps=d, post_steps=d)
        out['ce'][f'{d}+{d}'] = dev_ce(suite)
        print(f'{path.stem}: trained {trained[0]}+{trained[1]}, evaluated {d}+{d}: CE {out["ce"][f"{d}+{d}"]:.3f}', flush=True)
    out['seconds'] = time.time() - started
    Path(a.output).write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
