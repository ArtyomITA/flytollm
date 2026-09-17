"""Serial resume probes on the actual three trained2000-update checkpoints."""
import json
import pretrain_night_queue as q


def main():
    ready=json.loads((q.ROOT/'results/phase35_completion_live.json').read_text())
    assert ready['status']=='ready_for_night_launch', 'Finish replica/audit first; one GPU worker'
    q.STATE=q.ROOT/'results/trained_resume_smokes_live.json'
    if q.STATE.exists():raise RuntimeError('Trained resume smoke state already exists')
    state=dict(status='starting',completed=[]);q.write_state(state)
    try:
        for key in ('t10','h10','h5'):
            source=q.ROOT/'results'/f'pretrain_pilot_{key}.latest.pt'
            result=q.run_job(state,f'resume_candidate_{key}_smoke','verify_resume_after_probe',['--checkpoint',source],600)
            assert result['restored_step']==2000 and result['next_step']==2001
        state.update(status='completed');q.write_state(state)
        path=q.ROOT/'results/phase35_4_final_review.json';review=json.loads(path.read_text())
        review['trained_checkpoint_resume_smokes']=state['completed']
        path.write_text(json.dumps(review,indent=2),encoding='utf8')
    except Exception as exc:
        state.update(status='failed',error=str(exc));q.write_state(state);raise


if __name__=='__main__':main()
