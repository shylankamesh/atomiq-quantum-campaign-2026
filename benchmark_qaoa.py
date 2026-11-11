#!/usr/bin/env python3
"""Compare 10-qubit X and cardinality-preserving XY QAOA on identical inputs/seeds.

This produces simulated statevector distribution statistics, not hardware results.
The final six IDs are STILL selected by the exact certified classical solver.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from quantum_campaign.stage2 import energy_table, exact_solve, optimize_qaoa, state_probabilities
from quantum_campaign.advanced_qaoa import optimize_xy, xy_state_probabilities, outcome_metrics


def run(source, *, seeds=(11,22,33), layers=2, iterations=12, cvar=1.0):
    source=Path(source)
    stage2=json.loads(source.read_text(encoding='utf-8'))
    records=stage2['inputs']
    costs=np.array([r['cost'] for r in records]); caps=np.array([r['capacity'] for r in records]);
    rewards=np.array([r['reward'] for r in records]);
    B=float(stage2['total_budget']); H=float(stage2['total_capacity'])
    best,info=exact_solve(costs,caps,rewards,B,H)
    energies=energy_table(costs,caps,rewards,B,H)
    optimum_mask=sum(1<<i for i in best)
    results=[]
    for seed in seeds:
        angles,expectation=optimize_qaoa(energies,10,seed=seed,layers=layers,maxiter=iterations,cvar_alpha=cvar)
        p=state_probabilities(angles,energies,10)
        results.append(dict(mixer='x',seed=int(seed),penalized_energy=expectation,
                            **outcome_metrics(p,costs,caps,rewards,B,H,optimum_mask)))
        angles,expectation,_=optimize_xy(energies,10,6,seed=seed,layers=layers,maxiter=iterations,cvar_alpha=cvar)
        p=xy_state_probabilities(angles,energies,10,6)
        results.append(dict(mixer='xy',seed=int(seed),penalized_energy=expectation,
                            **outcome_metrics(p,costs,caps,rewards,B,H,optimum_mask)))
    frame=pd.DataFrame(results)
    summary = {m: {col: {'mean':float(part[col].mean()),
                          'std':float(part[col].std(ddof=1)) if len(part)>1 else 0.}
                   for col in ['feasible_probability','cardinality_probability',
                               'optimum_probability']}
               for m,part in frame.groupby('mixer')}
    payload=dict(mode='REFERENCE STATEVECTOR (NOT QISKIT AER OR QUANTUM HARDWARE)',
                 input_file=str(source),num_seeds=len(seeds),seeds=list(seeds),
                 layers=layers,iterations=iterations,cvar_lower_tail_alpha=cvar,
                 exact_objective=info['objective'],
                 conclusion_caution='XY always preserves exact count 6, but may not outperform X in feasible mass or objective on every dataset.',
                 summary=summary, records=results)
    return frame,payload


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage2-json',type=Path,default=Path('example/demo_outputs/stage2_selection.json'))
    ap.add_argument('--out',type=Path,default=None)
    ap.add_argument('--seeds',type=int,nargs='+',default=[11,22,33])
    ap.add_argument('--layers',type=int,default=2)
    ap.add_argument('--iterations',type=int,default=12)
    ap.add_argument('--cvar',type=float,default=1.0,help='Lower-tail CVaR alpha; 1.0 is mean')
    args=ap.parse_args()
    frame,payload=run(args.stage2_json,seeds=args.seeds,layers=args.layers,iterations=args.iterations,cvar=args.cvar)
    out=args.out or args.stage2_json.parent
    out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out/'qaoa_multiseed.csv',index=False)
    (out/'qaoa_multiseed.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8.4,4.8))
        styles={'x':0,'xy':1}
        for mixer,col in styles.items():
            one=frame[frame.mixer==mixer]
            mean=one.feasible_probability.mean()
            std=one.feasible_probability.std(ddof=1) if len(one)>1 else 0
            ax.bar(col,100*mean, yerr=100*std, capsize=6, width=.56)
            jitter=np.linspace(-.10,.10,len(one))
            ax.scatter(col+jitter,100*one.feasible_probability,s=24,zorder=3)
        ax.set_xticks([0,1],['Original X mixer','Exact-six XY mixer'])
        ax.set_ylabel('Probability mass on feasible subsets (%)')
        ax.set_title('Synthetic reference experiment — multi-seed comparison')
        ax.set_ylim(0,max(100*frame.feasible_probability.max()*1.2,1))
        ax.grid(axis='y',alpha=.15)
        fig.text(.5,.02,'Simulation only • Both budget & capacity enforced in feasibility check',
                 fontsize=9,ha='center')
        fig.tight_layout(rect=[0,.05,1,1])
        fig.savefig(out/'qaoa_multiseed.png',dpi=150)
        plt.close(fig)
    except ImportError:
        pass  # CLI JSON/CSV benchmark remains complete without matplotlib
    print(frame.to_string(index=False));print('\nWrote multi-seed benchmark to',out)
if __name__=='__main__':
    main()
