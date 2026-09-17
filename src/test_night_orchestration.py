"""CPU-only integration tests for unattended scheduling and optional failures."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pretrain_night_queue as q


def result(key, update=2000, ce=5.0):
    return dict(repeated_dev=[dict(ce=ce)] * 2, updates=update, targets=update * 26,
                artificial_initial_sha256='same', stop_reason='update_budget',
                checkpoint=f'results/{key}-{update}.pt', checkpoint_sha256='testhash', parameters=100, mean_step_s=.7,
                initial_dev=dict(ce=8), curve=[dict(baseline={})], generation=[])


class NightTests(unittest.TestCase):
    def test_nonfinite_and_unfair_comparison_rejected(self):
        with self.assertRaises(AssertionError): q.score(result('bad', ce=float('nan')))
        with self.assertRaises(AssertionError):
            q.choose_night_pair(dict(t10=result('t10'), h10=result('h10', 8000)))

    def test_optional_failure_does_not_prevent_remaining_probes(self):
        state=dict(completed=[])
        with patch.object(q, 'run_job', return_value=None) as run, patch.object(q, 'write_state'):
            self.assertFalse(q.optional_suite(state, Path('unchanged.pt'), 'probe'))
        self.assertEqual(run.call_count, 5)
        self.assertFalse(state['analysis']['ok'])

    def test_explicit_stop_survives_optional_flag(self):
        with patch.object(q, 'run_job', side_effect=InterruptedError('stop')), patch.object(q, 'write_state'):
            with self.assertRaises(InterruptedError): q.optional_suite(dict(completed=[]), Path('x.pt'), 'probe')

    def test_malformed_analysis_input_is_optional(self):
        state=dict(completed=[],context_reset_delta_ce=3.7)
        with patch.object(q, 'run_job', return_value={}), patch.object(q, 'write_state'):
            q.optional_suite(state, Path('unchanged.pt'), 'probe')
        self.assertFalse(state['analysis']['ok'])
        self.assertNotIn('context_reset_delta_ce',state)

    def test_existing_probe_output_preserved(self):
        with tempfile.TemporaryDirectory(dir=q.ROOT/'.runtime-tmp') as folder:
            root=Path(folder); (root/'results').mkdir(); output=root/'results/existing.json'
            output.write_text('preserve me')
            with patch.object(q, 'ROOT', root), patch.object(q, 'STOP', root/'STOP'), patch.object(q, 'write_state'):
                self.assertIsNone(q.run_job(dict(completed=[]), 'existing', 'unused', [], 1, optional=True))
            self.assertEqual(output.read_text(), 'preserve me')

    def test_optional_spawn_failure_logged(self):
        with tempfile.TemporaryDirectory(dir=q.ROOT/'.runtime-tmp') as folder:
            root=Path(folder); (root/'results').mkdir(); state=dict(completed=[])
            with patch.object(q, 'ROOT', root), patch.object(q, 'STOP', root/'STOP'), patch.object(q, 'write_state'), patch.object(q.subprocess, 'Popen', side_effect=OSError('injected spawn failure')):
                self.assertIsNone(q.run_job(state, 'missing', 'unused', [], 1, optional=True))
            self.assertIn('injected', state['completed'][0]['error'])

    def test_full_night_sequence_and_exact_probe_resume_source(self):
        for scores, order in [((6, 5.8, 5.7), ['h5', 'h10']), ((5.5, 5.8, 5.7), ['t10', 'h10'])]:
            with self.subTest(scores=scores), tempfile.TemporaryDirectory(dir=q.ROOT/'.runtime-tmp') as folder:
                root=Path(folder); (root/'results').mkdir(); launched=[]
                checks=['pretrain_smoke_t10','pretrain_resume_smoke_t10','pretrain_smoke_h5','pretrain_resume_smoke_h5',
                        'resume_after_suite_smoke','phase35_audit_harness_smoke']
                checks += [f'resume_candidate_{key}_smoke' for key in ('t10','h10','h5')]
                checks += [name for name, _ in q.probe_plan('x', 'graph_suite_smoke_h5', smoke=True)]
                for name in checks: (root/'results'/f'{name}.json').write_text(json.dumps(dict(ok=True)))
                for key, ce in zip(('t10','h10','h5'), scores):
                    (root/'results'/f'pretrain_pilot_{key}.json').write_text(json.dumps(dict(ok=True, result=result(key,ce=ce))))
                (root/'results/phase35_4_compare_live.json').write_text(json.dumps(dict(status='completed_waiting_user')))
                (root/'results/phase35_4_final_review.json').write_text(json.dumps(dict(ok=True,night_launch_authorized_by_user=True,selected=order[0],checkpoint_sha256='testhash')))
                def run(state, name, module, args, timeout, optional=False):
                    launched.append((name, args))
                    update=int(args[args.index('--updates')+1]); r=result(name,update,ce=4.9)
                    if update==0:r['stop_reason']='user_stop_file'
                    return r
                with patch.object(q, 'ROOT', root), patch.object(q, 'STATE', root/'state.json'), patch.object(q, 'run_job', side_effect=run), patch.object(q, 'optional_suite', return_value=False) as suite, patch.object(q.ctypes.windll.kernel32, 'SetThreadExecutionState', return_value=1), patch('sys.argv', ['queue', '--from-comparison']):
                    q.main()
                self.assertEqual([name for name, _ in launched[:2]], [f'pretrain_night8000_{key}' for key in order])
                self.assertEqual([name for name, _ in launched[2:]], ['pretrain_to20000', 'pretrain_continuous'])
                source=suite.call_args.args[1]
                resume_args=launched[-1][1]
                self.assertEqual(source, resume_args[resume_args.index('--resume')+1])
                final=json.loads((root/'results/pretrain_actual_night_live.json').read_text())
                self.assertEqual(final['reason'],'user_stop_file')
                self.assertFalse(final['probe_20000_ok'])


if __name__ == '__main__':
    outcome=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(NightTests))
    (q.ROOT/'results/night_control_smoke.json').write_text(json.dumps(dict(ok=outcome.wasSuccessful(),
        tests=outcome.testsRun,failures=len(outcome.failures),errors=len(outcome.errors),cpu_only=True),indent=2))
    sys.exit(0 if outcome.wasSuccessful() else 1)
