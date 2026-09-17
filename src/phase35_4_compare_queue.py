"""Finish authorized phase3.5+4 comparisons; never start long pretraining."""
import json,time
from pathlib import Path
import pretrain_night_queue as q

def main():
    q.STATE=q.ROOT/'results/phase35_4_compare_live.json'
    if q.STATE.exists():raise RuntimeError('Comparison queue already exists')
    state=dict(status='waiting_current_pilot',current='pretrain_pilot_t10',completed=[],long_pretraining_authorized=False)
    q.write_state(state)
    first=q.ROOT/'results/pretrain_pilot_t10.json'; deadline=time.monotonic()+10800
    while not first.exists():
        if q.STOP.exists():raise InterruptedError('User stopped experiments')
        if time.monotonic()>deadline:raise TimeoutError('Existing pilot did not finish')
        time.sleep(10)
    try:
        data=json.loads(first.read_text())
        if not data.get('ok'):raise RuntimeError('Current tied10 pilot failed')
        results={'t10':data['result']}
        state['completed'].append(dict(name='pretrain_pilot_t10',ok=True));q.write_state(state)
        for key,threshold in [('h10',10),('h5',5)]:
            results[key]=q.run_job(state,f'pretrain_pilot_{key}','pretrain_resumable',
                                  ['--threshold',threshold,'--head','separate','--updates',2000],7200)
        selected=q.choose(results)
        state.update(status='completed_waiting_user',recommended=selected,
                     comparison={k:dict(ce=q.score(v),targets=v['targets'],updates=v['updates'],parameters=v['parameters'],
                                         mean_step_s=v['mean_step_s'],baseline=v['curve'][-1]['baseline'],
                                         checkpoint=v['checkpoint'],generation=v['generation']) for k,v in results.items()},
                     next='Present phase3.5+4 results; long pretraining only after explicit bedtime launch from user.')
        q.write_state(state);print(json.dumps(dict(completed=True,recommended=selected,long_pretraining_started=False)),flush=True)
    except Exception as exc:
        state.update(status='failed',error=str(exc));q.write_state(state);raise

if __name__=='__main__':main()
