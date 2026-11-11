#!/usr/bin/env python3
"""Atomiq campaign runner. Real runs require the actual organizer data and limits.

Demo outputs are isolated from repository root to prevent accidental submission.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from quantum_campaign import __version__
from quantum_campaign.data import generate_demo, resolve_target, validate_inputs, load_organizer_workbook, join_metadata, normalize_target
from quantum_campaign.stage1 import train_and_predict
from quantum_campaign.stage2 import solve as stage2_solve
from quantum_campaign.stage3 import simulate as stage3_simulate


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--demo', action='store_true', help='Use explicitly synthetic generated data; outputs go in example/demo_outputs/')
    p.add_argument('--train', type=Path, help='Actual organizers training file, CSV or XLSX')
    p.add_argument('--workbook', type=Path, help='Official participant workbook with all five named sheets')
    p.add_argument('--resume-stage1', action='store_true', help='Reuse frozen verified Stage1 CSV and diagnostics after matching input hashes and seeds')
    p.add_argument('--stage2-data', type=Path, help='Separate CSV/XLSX metadata joined by customer ID')
    p.add_argument('--test', type=Path, help='Actual organizers test file, CSV or XLSX')
    p.add_argument('--out', type=Path, default=Path('.'), help='Result output folder for REAL dataset')
    p.add_argument('--id-column', default='customer_id')
    p.add_argument('--target', default=None, help='Training binary label column')
    p.add_argument('--cost-column', default='contact_cost')
    p.add_argument('--capacity-column', default='capacity_units')
    p.add_argument('--value-column', default=None, help='Needed for revenue/profit objectives')
    p.add_argument('--objective', choices=['expected_conversions','expected_revenue','expected_profit'],default='expected_profit')
    p.add_argument('--budget', type=float, help='MANDATORY official maximum budget for real run')
    p.add_argument('--capacity', type=float, help='MANDATORY official maximum capacity for real run')
    p.add_argument('--engine', choices=['auto','qiskit','numpy'],default='auto')
    p.add_argument('--qml-backend', choices=['auto','native','qiskit_ml'], default='auto',
                   help='Qiskit ML FidelityQuantumKernel+QSVC or fast 4-qubit fidelity reference')
    p.add_argument('--qaoa-mixer', choices=['xy','x'], default='xy',
                   help='XY fixes exact cardinality to six; x is previous unconstrained mixer')
    p.add_argument('--compare-mixers', action='store_true', help='Run XY and X with same optimization budget')
    p.add_argument('--seed', type=int,default=42)
    p.add_argument('--qaoa-layers', type=int,default=2)
    p.add_argument('--qaoa-iterations', type=int,default=12)
    p.add_argument('--qaoa-cvar', type=float, default=1.0, help='QAOA lower-tail CVaR objective alpha in (0,1], 1=mean')
    p.add_argument('--qaoa-shots', type=int,default=4096)
    p.add_argument('--simulation-shots', type=int,default=8192)
    p.add_argument('--mc-iterations',type=int,default=100000)
    p.add_argument('--readout-error',type=float,default=.025)
    return p


def table(path):
    if path.suffix.lower() in ('.xlsx','.xls'):
        return pd.read_excel(path)
    if path.suffix.lower()=='.csv':
        return pd.read_csv(path)
    raise ValueError('Training/test files must be CSV or XLSX')


def md5(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path,obj):
    def to_json(v):
        if isinstance(v,(np.integer,)): return int(v)
        if isinstance(v,(np.floating,)): return float(v)
        if isinstance(v,(np.bool_,)): return bool(v)
        if isinstance(v,np.ndarray): return v.tolist()
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(obj,indent=2,default=to_json,allow_nan=False),encoding='utf-8')


def make_figures(pred, s2, s3, diagnostics, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(8,4.1))
    ax.plot(pred['rank'],pred['predicted_probability'],linewidth=2)
    ax.scatter(pred['rank'].iloc[:10],pred['predicted_probability'].iloc[:10],s=24)
    ax.axvline(10,linestyle='--',linewidth=1)
    ax.set(xlabel='Predicted rank',ylabel='Subscription probability',title='Stage 1 — Customer ranking (top 10 marked)')
    ax.set_ylim(0,1)
    fig.tight_layout();fig.savefig(out/'stage1_ranking.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    ids=s2['candidate_ids']; ps=[x['p'] for x in s2['inputs']]
    selected=set(s2['selected_customer_ids'])
    labels=[('● ' if x in selected else '  ')+x for x in ids]
    ax.barh(range(10),ps)
    ax.set_yticks(range(10),labels);ax.invert_yaxis()
    ax.set(xlabel='Predicted conversion probability',title='Stage 2 — Top ten candidates (● = selected)')
    fig.tight_layout();fig.savefig(out/'stage2_selection.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    vals=[s3['ideal_absolute_error'],s3['noisy_absolute_error'],
          s3['mitigated_absolute_error'],s3['monte_carlo_absolute_error']]
    names=['Quantum ideal','Noisy readout','Readout corrected','Monte Carlo']
    ax.bar(names,vals)
    ax.set_ylabel('Absolute probability error vs analytical truth')
    ax.set_title('Stage 3 — Error comparison (lower is better)')
    ax.tick_params(axis='x',labelrotation=16)
    for i,v in enumerate(vals):
        ax.text(i,v+max(vals)*.035,f'{v:.5f}',ha='center',va='bottom',fontsize=10)
    ax.set_ylim(0,max(vals)*1.28 if max(vals)>0 else 1e-4)
    fig.tight_layout();fig.savefig(out/'stage3_comparison.png',dpi=150);plt.close(fig)


def main():
    import time
    started = time.perf_counter()
    args = parser().parse_args()
    if args.qaoa_layers < 1 or args.qaoa_iterations < 1 or args.qaoa_shots < 1 or args.simulation_shots < 1 or args.mc_iterations < 1:
        raise ValueError('Layers, iterations, shots and Monte Carlo iterations must be positive')
    if not 0 < args.qaoa_cvar <= 1:
        raise ValueError('--qaoa-cvar must be in (0,1]')
    if not 0 <= args.readout_error < .5:
        raise ValueError('Readout error must be in [0,0.5)')
    if args.demo:
        if args.train or args.test:
            raise ValueError('Do not mix --demo with real training/test files')
        train,test=generate_demo(args.seed)
        budget,capacity=100.,24.
        out=Path('example/demo_outputs')
        synthetic=True
        args.target='subscribed'
        args.cost_column='contact_cost'
        args.capacity_column='capacity_units'
        if args.objective in ('expected_profit','expected_revenue'):
            args.value_column='expected_value'
    elif args.workbook:
        if args.train or args.test or args.stage2_data:
            raise ValueError('Use --workbook alone, or separate input files')
        train,test,budget,capacity=load_organizer_workbook(args.workbook)
        args.target='y'
        args.value_column='expected_deposit_value'
        if args.objective != 'expected_profit':
            raise ValueError('Official workbook requires expected_profit = p*deposit_value-contact_cost')
        if args.budget is not None or args.capacity is not None:
            raise ValueError('Official limits are read from workbook, not overridden')
        out=args.out
        synthetic=False
    else:
        if not args.train or not args.test or args.budget is None or args.capacity is None:
            raise ValueError('REAL submission requires --train, --test, --budget and --capacity. Read the organizers briefing for the correct values.')
        train,test=table(args.train),table(args.test)
        budget,capacity=args.budget,args.capacity
        out=args.out
        synthetic=False
        args.target=resolve_target(train,args.target)
        train[args.target]=normalize_target(train[args.target])
        if len(train)!=300:
            raise ValueError('Official training dataset must have 300 rows')
        if args.stage2_data:
            test=join_metadata(test,table(args.stage2_data),args.id_column)
    if budget<=0 or capacity<=0:
        raise ValueError('Budget/capacity must be positive')
    feature_cols=validate_inputs(train,test,id_col=args.id_column,target=args.target,
                                 cost_col=args.cost_column,capacity_col=args.capacity_column,
                                 value_col=args.value_column,expected_test_rows=100)
    out.mkdir(parents=True,exist_ok=True)
    if args.resume_stage1:
        if not args.workbook:
            raise ValueError('--resume-stage1 requires the original organizer workbook')
        old=json.loads((out/'run_manifest.json').read_text())
        integrity=json.loads((out/'validation_manifest.json').read_text())
        if old['organizer_workbook_hash']!=md5(args.workbook) or old['seed']!=args.seed or old['qml_backend']!=args.qml_backend:
            raise ValueError('Frozen Stage1 inputs/settings do not match')
        for name in ['stage1_predictions.csv','stage1_validation.json']:
            if md5(out/name)!=integrity['output_sha256'][name]:
                raise ValueError(f'Frozen Stage1 hash mismatch: {name}')
        stage1=pd.read_csv(out/'stage1_predictions.csv',dtype={'customer_id':str},float_precision='round_trip')
        diagnostics=json.loads((out/'stage1_validation.json').read_text())
    else:
        stage1,diagnostics=train_and_predict(train,test,feature_cols,id_col=args.id_column,
                                            target=args.target, seed=args.seed,engine=args.engine,
                                            qml_backend=args.qml_backend,model_path=out/'fitted_model.joblib')
    # Only persisted predictions enter later stages. No ground-truth labels are used.
    predictions=stage1.copy()
    if not np.all((predictions['predicted_probability']>=0)&(predictions['predicted_probability']<=1)):
        raise AssertionError('Invalid predicted probability')
    stage1.to_csv(out/'stage1_predictions.csv',index=False,float_format='%.17g')
    stage1=pd.read_csv(out/'stage1_predictions.csv',dtype={'customer_id':str},float_precision='round_trip')
    test_meta=test.drop(columns=[args.target],errors='ignore').copy()
    test_meta[args.id_column] = test_meta[args.id_column].astype(str)
    top10=stage1.iloc[:10].merge(test_meta,left_on='customer_id',right_on=args.id_column,
                                validate='one_to_one',sort=False)
    if args.id_column != 'customer_id':
        top10=top10.drop(columns=[args.id_column])
    s2=stage2_solve(top10,budget=budget,capacity_limit=capacity,
                   cost_col=args.cost_column,capacity_col=args.capacity_column,
                   value_col=args.value_column,objective=args.objective,seed=args.seed,
                   shots=args.qaoa_shots,engine=args.engine,qaoa_layers=args.qaoa_layers,
                   qaoa_iterations=args.qaoa_iterations,
                   qaoa_mixer=args.qaoa_mixer,compare_mixers=args.compare_mixers,
                   qaoa_cvar=args.qaoa_cvar)
    # If using a reference NumPy simulator, do not report Qiskit/Aer execution.
    save_json(out/'stage2_selection.json',s2)
    p_by_id=stage1.set_index('customer_id')['predicted_probability'].to_dict()
    customer_ids=s2['selected_customer_ids']
    selected_probs=[float(p_by_id[c]) for c in customer_ids]
    s3=stage3_simulate(customer_ids,selected_probs,shots=args.simulation_shots,
                       mc_iterations=args.mc_iterations,readout_error=args.readout_error,
                       seed=args.seed,engine=args.engine)
    save_json(out/'stage3_simulation.json',s3)
    save_json(out/'stage1_validation.json',diagnostics)
    packages={}
    from importlib.metadata import version,PackageNotFoundError
    for pkg in ('numpy','pandas','scikit-learn','scipy','qiskit','qiskit-aer','qiskit-machine-learning'):
        try: packages[pkg]=version(pkg)
        except PackageNotFoundError: packages[pkg]='not installed'
    config=dict(demo=synthetic,version=__version__,seed=args.seed,engine_requested=args.engine,
                stage1_quantum_feature_state_engine=args.engine,
                stage2_engine_used=s2['quantum_engine'],stage3_engine_used=s3['quantum_engine'],
                features=feature_cols,id_column=args.id_column,target_column=args.target,
                qml_backend=diagnostics['quantum_backend'],qaoa_mixer=args.qaoa_mixer,
                qaoa_cvar=args.qaoa_cvar,
                cost_column=args.cost_column,capacity_column=args.capacity_column,
                value_column=args.value_column,objective=args.objective,budget=budget,capacity=capacity,
                python=platform.python_version(),packages=packages,
                seeds={'split':args.seed,'SVC':args.seed,'outer_optimizer':args.seed,
                       'transpiler':args.seed,'qaoa_simulator':args.seed,
                       'stage3_ideal':args.seed+1,'stage3_noisy':args.seed+2,
                       'monte_carlo':args.seed+501},
                training_file_hash=None if synthetic else md5(args.workbook or args.train),
                test_file_hash=None if synthetic else md5(args.workbook or args.test),
                organizer_workbook_hash=md5(args.workbook) if args.workbook else None,
                organizer_source='https://drive.google.com/file/d/1GV6QsrWGTyPStOLR_hjlGo2ZHq75C_4K/view' if args.workbook else None,
                metadata_file_hash=md5(args.stage2_data) if args.stage2_data else None,
                cpu=platform.processor(),platform=platform.platform(),
                runtime_seconds=time.perf_counter()-started,
                stage1_reused=args.resume_stage1, stage1_runtime_seconds=diagnostics['runtime_seconds'],
                disclaimer=('SYNTHETIC DEMONSTRATION; NOT VALID AS AN OFFICIAL SUBMISSION'
                            if synthetic else 'Real dataset run; verify objective/constraints against organizer briefing'))
    save_json(out/'run_manifest.json',config)
    save_json(out/'validation_manifest.json',{'input_sha256': config['organizer_workbook_hash'],
        'output_sha256':{name:md5(out/name) for name in ['stage1_predictions.csv','stage2_selection.json','stage3_simulation.json','stage1_validation.json']}})
    make_figures(stage1,s2,s3,diagnostics,out)
    print('\n'+'='*68)
    print('ATOMIQ HYBRID QUANTUM CAMPAIGN', '(SYNTHETIC DEMO)' if synthetic else '(REAL INPUTS)')
    print('Stage 1: 100 predictions; QML:',diagnostics['quantum_backend'],
          'hybrid weight:',diagnostics['quantum_weight'],
          'validation Brier:',round(diagnostics['hybrid_validation_metrics']['brier'],5))
    print('Stage 2: selected IDs:',', '.join(customer_ids))
    print('Stage 2: objective:',round(s2['objective_value'],6),'QAOA mixer:',args.qaoa_mixer,
          'QAOA gap:',s2['qaoa_measured_optimality_gap'])
    print('Stage 3: ideal measurement P(>=3):',round(s3['quantum_estimated_probability'],6))
    print('Stage 3: exact analytical P(>=3):',round(s3['exact_analytical_probability'],6))
    print('Engines:',s2['quantum_engine'],s3['quantum_engine'])
    print('All results saved in:',out.resolve())
    if synthetic:
        print('WARNING: DEMO RESULTS ARE SYNTHETIC AND MUST NOT BE SUBMITTED AS REAL RESULTS')
    print('='*68)

if __name__=='__main__':
    main()
