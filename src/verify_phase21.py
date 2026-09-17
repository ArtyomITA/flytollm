"""Bounded CPU regression suite; persistent result and memory observations."""
import gc
import json
from pathlib import Path
import time
import unittest
from bench_runtime import memory
from prepare_text import sha


if __name__=='__main__':
    pre=memory()
    if pre['available_gib']<1.75: raise RuntimeError('Insufficient host RAM for verification')
    begin=time.perf_counter()
    suite=unittest.defaultTestLoader.loadTestsFromNames(['test_fly_core','test_core_api','test_text_pipeline'])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    gc.collect()
    report=dict(ok=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                elapsed_s=time.perf_counter()-begin,pre_memory=pre,post_memory=memory(),
                source_sha256={p:sha(p) for p in ['fly_core.py','test_fly_core.py','test_core_api.py','prepare_text.py',
                                                 'text_dataset.py','test_text_pipeline.py']},device='CPU')
    Path('results/phase21_cpu_tests.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    raise SystemExit(not result.wasSuccessful())
