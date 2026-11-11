"""Additional experiments on frozen predictions; never retrains or tunes Stage 1."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from run_pipeline import save_json,md5
from quantum_campaign.stage2 import solve, exact_solve
from quantum_campaign.stage3 import exact_tail_probability, conversion_circuit
from quantum_campaign.qubo import build_qubo
from quantum_campaign.qml_qiskit_ml import parameterized_feature_map
from quantum_campaign.circuit_metrics import basis_metrics


def main():
    root=Path('.')
    import sys
    if '--plots-only' in sys.argv:
        render_evidence()
        return
    s2=json.loads((root/'stage2_selection.json').read_text())
    s3=json.loads((root/'stage3_simulation.json').read_text())
    manifest=json.loads((root/'run_manifest.json').read_text())
    pred=pd.read_csv('stage1_predictions.csv',dtype={'customer_id':str},float_precision='round_trip')
    workbook=root/'data/bank_campaign_participant_release.xlsx'
    if md5(workbook)!=manifest['organizer_workbook_hash']:
        raise ValueError('Organizer workbook changed')
    meta=pd.read_excel(workbook,sheet_name='Stage2_Data',dtype={'customer_id':str})
    top=pred.iloc[:10].merge(meta,on='customer_id',validate='one_to_one',sort=False)
    inputs=s2['inputs']; c=np.array([x['cost'] for x in inputs]); a=np.array([x['capacity'] for x in inputs]); r=np.array([x['reward'] for x in inputs])
    save_json(root/'qubo_coefficients.json',build_qubo(c,a,r,25,22))
    pd.DataFrame(s2['enumeration']).to_csv('stage2_all_210_subsets.csv',index=False)
    trials=[]
    for depth in (1,2):
        for seed in (11,22,33):
            for mixer in ('x','xy'):
                print('QAOA trial',depth,seed,mixer,flush=True)
                start=time.perf_counter()
                result=solve(top,budget=25,capacity_limit=22,cost_col='contact_cost',capacity_col='capacity_units',value_col='expected_deposit_value',objective='expected_profit',seed=seed,shots=2048,engine='qiskit',qaoa_layers=depth,qaoa_iterations=6,qaoa_mixer=mixer)
                trials.append(dict(depth=depth,seed=seed,mixer=mixer,shots=2048,optimizer_maxiter=6,
                    runtime_seconds=time.perf_counter()-start,feasible_fraction=result['measured_feasible_fraction'],
                    cardinality_six_fraction=result['measured_cardinality_six_fraction'],
                    exact_optimum_fraction=result['measured_exact_optimum_fraction'],
                    best_objective=result['qaoa_best_measured_objective'],gap=result['qaoa_measured_optimality_gap'],
                    circuit=result['circuit'],counts=result['qaoa_measurement_counts']))
    save_json(root/'qaoa_repeated_trials.json',trials)
    pd.DataFrame([{k:v for k,v in x.items() if k not in ('circuit','counts')} for x in trials]).to_csv('qaoa_repeated_trials.csv',index=False)
    rng=np.random.default_rng(8042)
    probs=pred.predicted_probability.to_numpy(); ids=pred.customer_id.tolist()
    base=set(s2['selected_customer_ids']); perturb=[]
    data=meta.set_index('customer_id')
    for _ in range(200):
        pp=np.clip(probs+rng.normal(0,.01,len(probs)),0,1)
        order=sorted(range(len(ids)),key=lambda i:(-pp[i],ids[i]))[:10]
        candidates=[ids[i] for i in order]; rows=data.loc[candidates]
        try:
            chosen,info=exact_solve(rows.contact_cost.to_numpy(),rows.capacity_units.to_numpy(),pp[order]*rows.expected_deposit_value.to_numpy()-rows.contact_cost.to_numpy(),25,22)
            selected=[candidates[i] for i in chosen]
            perturb.append(dict(selected_ids=selected,overlap_with_final=len(base&set(selected)),
                expected_net_value=info['objective'],tail_probability=exact_tail_probability(pp[np.array(order)[list(chosen)]])))
        except ValueError:
            perturb.append(dict(infeasible=True))
    tails=[x['tail_probability'] for x in perturb if 'tail_probability' in x]
    save_json(root/'sensitivity_analysis.json',dict(seed=8042,trials=200,perturbation='Independent N(0,0.01^2) probability perturbations clipped to [0,1]; rerank all 100, reselect six',
        scope='Illustrative sensitivity only, NOT bootstrap confidence intervals or calibrated prediction uncertainty',
        unchanged_selection_fraction=sum(set(x.get('selected_ids',[]))==base for x in perturb)/200,
        mean_selection_overlap=float(np.mean([x['overlap_with_final'] for x in perturb if 'overlap_with_final' in x])),
        tail_probability_range=[min(tails),max(tails)],results=perturb))
    render_evidence()


def render_evidence():
    root=Path('.')
    s3=json.loads((root/'stage3_simulation.json').read_text())
    feature=parameterized_feature_map(4)
    fm=basis_metrics(feature,42)
    fm.update(logical_qubits=4,logical_depth=feature.depth(),fidelity_shots=1024,sampler='Qiskit StatevectorSampler',seed=42)
    save_json(root/'stage1_circuit_metrics.json',fm)
    (root/'figures').mkdir(exist_ok=True)
    (root/'figures/quantum_feature_map.txt').write_text(str(feature.draw(output='text')),encoding='utf-8')
    (root/'figures/stage3_circuit.txt').write_text(str(conversion_circuit(s3['probabilities']).draw(output='text')),encoding='utf-8')
    # Qiskit's own diagram renderer represents the actual executed circuit.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name,circuit in [('quantum_feature_map',feature),('stage3_circuit',conversion_circuit(s3['probabilities']))]:
        fig=circuit.draw(output='mpl',fold=-1)
        fig.savefig(root/f'figures/{name}.png',dpi=160,bbox_inches='tight'); plt.close(fig)
    d=json.loads((root/'stage1_validation.json').read_text())
    fig,ax=plt.subplots(figsize=(7,4))
    for name,bins in d['calibration'].items():
        ax.plot([b['mean_probability'] for b in bins],[b['observed_fraction'] for b in bins],marker='o',label=name)
    ax.plot([0,1],[0,1],'k--',linewidth=1); ax.set(xlabel='Mean predicted probability',ylabel='Observed subscription frequency',title='Training OOF calibration (five fixed bins)'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(root/'figures/calibration.png',dpi=160); plt.close(fig)


if __name__=='__main__': main()
