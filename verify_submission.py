#!/usr/bin/env python3
"""Independently check consistency of the Atomiq three-stage output bundle.

Fail closed on synthetic data and non-Qiskit engines unless --allow-demo.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations, product
import hashlib


def check(out:Path, allow_demo=False):
    paths=[out/'stage1_predictions.csv', out/'stage2_selection.json',
           out/'stage3_simulation.json', out/'run_manifest.json']
    missing=[p.name for p in paths if not p.exists()]
    if missing: raise AssertionError(f'Missing files: {missing}')
    pred=pd.read_csv(paths[0],dtype={'customer_id':str})
    s2=json.loads(paths[1].read_text())
    s3=json.loads(paths[2].read_text())
    meta=json.loads(paths[3].read_text())
    check_assert=lambda ok,msg: (_ for _ in ()).throw(AssertionError(msg)) if not ok else None
    check_assert(not meta['demo'] or allow_demo,'DEMO data detected. Never submit generated demo IDs.')
    check_assert((s2['quantum_engine']=='qiskit' and s3['quantum_engine']=='qiskit') or allow_demo,
                 'Qiskit Aer was not used: run with --engine qiskit')
    check_assert(len(pred)==100,'Expected exactly 100 predictions')
    check_assert(set(['customer_id','predicted_probability','rank','is_top_10']).issubset(pred.columns),'Stage1 mandatory columns missing')
    check_assert(pred.customer_id.nunique()==100,'Duplicate IDs in Stage 1')
    check_assert(allow_demo or not pred.customer_id.str.startswith('DEMO_').any(),
                 'Synthetic DEMO_ IDs present in alleged real submission')
    check_assert(np.array_equal(pred['rank'].to_numpy(),np.arange(1,101)),'Ranks must be 1..100 in order')
    check_assert(pred.predicted_probability.notna().all() and pred.predicted_probability.between(0,1).all(),'Invalid probabilities')
    check_assert((pred.predicted_probability.diff().iloc[1:]<=1e-12).all(),'Ranking is not descending')
    check_assert(pred.is_top_10.map(lambda v:str(v).lower()=='true').sum()==10,'Exactly 10 top candidates required')
    top=pred.iloc[:10].customer_id.tolist()
    check_assert(top==s2['candidate_ids'],'Stage2 candidate IDs do not match Stage1 top ten')
    selected=s2['selected_customer_ids']
    check_assert(len(selected)==6 and len(set(selected))==6 and set(selected).issubset(set(top)),'Not 6 unique eligible selected IDs')
    check_assert(selected==s3['selected_customer_ids'],'Stage2 and Stage3 ID order inconsistent')
    probs=dict(zip(pred.customer_id,pred.predicted_probability))
    check_assert(np.allclose(s3['probabilities'],[probs[x] for x in selected],rtol=0,atol=1e-10), 'Stage3 p values changed from Stage1')
    check_assert(all(x['customer_id']==selected[x['qubit']] and x['qubit']==i for i,x in enumerate(s3['customer_to_qubit_mapping'])),'Qubit mapping inconsistent')
    check_assert(np.allclose(s3['ry_angles_radians'],2*np.arcsin(np.sqrt(s3['probabilities'])),atol=1e-10),'RY angles inconsistent')
    exact=sum(np.prod([p if bit else 1-p for p,bit in zip(s3['probabilities'],bits)]) for bits in product((0,1),repeat=6) if sum(bits)>=3)
    check_assert(abs(exact-s3['exact_analytical_probability'])<1e-11,'Stage3 analytical result inconsistent')
    check_assert(sum(s3['ideal_measurement_counts'].values())==s3['shots'],'Stage3 ideal shot counts do not sum to shots')
    check_assert(sum(s3['noisy_measurement_counts'].values())==s3['shots'],'Stage3 noisy shot counts do not sum to shots')
    check_assert(abs(s3['quantum_estimated_probability']-sum(v for k,v in s3['ideal_measurement_counts'].items() if k.count('1')>=3)/s3['shots'])<1e-10,'Stage3 probability inconsistent with counts')
    if 'qaoa_measurement_counts' in s2:
        qcounts=s2['qaoa_measurement_counts']
        check_assert(sum(qcounts.values())==s2['shots'],'Stage2 shots do not sum to declared count')
        check_assert(all(len(k)==10 and set(k)<={'0','1'} for k in qcounts),
                     'Stage2 measurement bitstrings malformed')
        measured_card=sum(v for k,v in qcounts.items() if k.count('1')==6)/s2['shots']
        check_assert(abs(measured_card-s2['measured_cardinality_six_fraction'])<1e-11,
                     'Stage2 Hamming-weight metric inconsistent with actual counts')
        if s2.get('qaoa_mixer')=='xy' and meta.get('engine_requested')!='numpy':
            # Both ideal Aer and reference XY are cardinality-preserving. With
            # real-device noise or noisy simulation this requirement may fail.
            check_assert(abs(measured_card-1.)<1e-10,
                         'Ideal XY mixer reported shot counts outside the six-subset space')
    inputs=s2['inputs']
    check_assert([item['customer_id'] for item in inputs]==top,'Stage2 input list not in Stage1 rank order')
    costs=np.array([x['cost'] for x in inputs]); caps=np.array([x['capacity'] for x in inputs]); rewards=np.array([x['reward'] for x in inputs])
    indices=tuple(top.index(x) for x in selected)
    check_assert(abs(s2['budget_used']-sum(costs[list(indices)]))<1e-8,'Reported cost inconsistent')
    check_assert(abs(s2['capacity_used']-sum(caps[list(indices)]))<1e-8,'Reported capacity inconsistent')
    check_assert(s2['budget_used']<=s2['total_budget']+1e-9 and s2['capacity_used']<=s2['total_capacity']+1e-9,'Constraint violation')
    check_assert(abs(s2['objective_value']-sum(rewards[list(indices)]))<1e-9,'Objective inconsistent')
    feasible=[t for t in combinations(range(10),6) if sum(costs[list(t)])<=s2['total_budget']+1e-8 and sum(caps[list(t)])<=s2['total_capacity']+1e-8]
    opt={'objective':max(sum(rewards[list(t)]) for t in feasible)}
    check_assert(abs(opt['objective']-s2['classical_bruteforce_optimum'])<1e-9,'Classical optimum incorrect')
    check_assert(abs(opt['objective']-s2['objective_value'])<1e-9,'Final choice is not classical optimum')
    check_assert(np.array_equal(pred.is_top_10.astype(str).str.lower().eq('true'),pred['rank']<=10),'Top-ten flags do not match ranks')
    ordered=pred.sort_values(['predicted_probability','customer_id'],ascending=[False,True],kind='stable')
    check_assert(ordered.customer_id.tolist()==pred.customer_id.tolist(),'Alphanumeric tie ordering invalid')
    check_assert(all(abs(item['p']-probs[item['customer_id']])<1e-14 for item in inputs),'Stage2 probabilities changed')
    check_assert(all(abs(item['probability']-probs[item['customer_id']])<1e-14 for item in s3['customer_to_qubit_mapping']),'Mapping probabilities changed')
    if not allow_demo:
        check_assert(s2['objective_type']=='expected_profit','Official objective must be expected net value')
    if s2['objective_type']=='expected_profit':
        check_assert(all(abs(item['reward']-(item['p']*item['expected_deposit_value']-item['cost']))<1e-10 for item in inputs),'Expected net value formula violated')
    measured=[]; feasible_shots=0
    for bits,freq in s2['qaoa_measurement_counts'].items():
        check_assert(isinstance(freq,int) and freq>=0,'Invalid QAOA count')
        idx=[i for i in range(10) if int(bits,2)&(1<<i)]
        if len(idx)==6 and costs[idx].sum()<=s2['total_budget']+1e-8 and caps[idx].sum()<=s2['total_capacity']+1e-8:
            measured.append(float(rewards[idx].sum())); feasible_shots+=freq
    best=max(measured) if measured else None
    check_assert((best is None and s2['qaoa_best_measured_objective'] is None) or (best is not None and abs(best-s2['qaoa_best_measured_objective'])<1e-9),'Measured quantum objective incorrect')
    if best is not None:
        check_assert(abs(s2['qaoa_measured_optimality_gap']-(opt['objective']-best))<1e-9,'Quantum gap incorrect')
    check_assert(abs(s2['measured_feasible_fraction']-feasible_shots/s2['shots'])<1e-12,'Feasible fraction incorrect')
    check_assert(abs(s2['final_optimality_gap']-(opt['objective']-s2['objective_value']))<1e-9,'Final gap incorrect')
    for key in ['ideal_measurement_counts','noisy_measurement_counts']:
        check_assert(all(len(bits)==6 and set(bits)<={'0','1'} and isinstance(freq,int) and freq>=0 for bits,freq in s3[key].items()),'Malformed Stage3 counts')
    noisy=sum(v for k,v in s3['noisy_measurement_counts'].items() if k.count('1')>=3)/s3['shots']
    check_assert(abs(noisy-s3['quantum_noisy_probability'])<1e-12,'Noisy probability inconsistent')
    check_assert(s3['threshold']==3 and s3['circuit']['logical_qubits']==6,'Incorrect stage3 threshold or qubits')
    for name in ['quantum_estimated_probability','quantum_noisy_probability','quantum_mitigated_probability','classical_monte_carlo_probability']:
        check_assert(np.isfinite(s3[name]) and 0<=s3[name]<=1,'Invalid Stage3 probability')
    for field,value in [('ideal_absolute_error',s3['quantum_estimated_probability']),('noisy_absolute_error',noisy),('mitigated_absolute_error',s3['quantum_mitigated_probability']),('monte_carlo_absolute_error',s3['classical_monte_carlo_probability'])]:
        check_assert(abs(s3[field]-abs(value-exact))<1e-12,'Error metric inconsistent')
    integrity=out/'validation_manifest.json'
    if integrity.exists():
        manifest=json.loads(integrity.read_text())
        for filename,digest in manifest['output_sha256'].items():
            check_assert(hashlib.sha256((out/filename).read_bytes()).hexdigest()==digest,f'Output hash mismatch: {filename}')
    if not allow_demo and meta.get('organizer_workbook_hash'):
        source=out/'data/bank_campaign_participant_release.xlsx'
        check_assert(source.exists(),'Organizer workbook absent')
        check_assert(hashlib.sha256(source.read_bytes()).hexdigest()==meta['organizer_workbook_hash'],'Organizer workbook hash mismatch')
        official=pd.read_excel(source,sheet_name='Stage1_Test',dtype={'customer_id':str})
        check_assert(set(official.customer_id)==set(pred.customer_id),'Predictions do not match official test IDs')
        operational=pd.read_excel(source,sheet_name='Stage2_Data',dtype={'customer_id':str}).set_index('customer_id')
        for item in inputs:
            row=operational.loc[item['customer_id']]
            check_assert(item['cost']==row.contact_cost and item['capacity']==row.capacity_units and item['expected_deposit_value']==row.expected_deposit_value,'Stage2 metadata differs from organizer workbook')
        check_assert(s2['total_budget']==25 and s2['total_capacity']==22,'Official constraints changed')
    return dict(status='PASS', mode='SYNTHETIC ONLY' if meta['demo'] else 'REAL-INPUT DATA',
                engine=s2['quantum_engine'], stage1_rows=100,stage2_selected=selected,
                stage3_probability=s3['quantum_estimated_probability'],
                stage3_exact=exact,verified_constraints=True, verified_customer_traceability=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=Path('.'))
    ap.add_argument('--allow-demo',action='store_true',help='Explicitly allow synthetic outputs and NumPy reference engines')
    args=ap.parse_args()
    result=check(args.out,args.allow_demo)
    print(json.dumps(result,indent=2))
