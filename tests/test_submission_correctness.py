from itertools import product, combinations
import numpy as np
import pandas as pd
import pytest
from quantum_campaign.data import normalize_target, join_metadata, load_organizer_workbook
from quantum_campaign.stage1 import rank_predictions, preprocessor
from quantum_campaign.stage2 import exact_solve, energy_table
from quantum_campaign.stage3 import conversion_circuit, exact_tail_probability, simulate
from quantum_campaign.qubo import build_qubo, qubo_energy, ising_energy
from quantum_campaign.advanced_qaoa import xy_qiskit_circuit
from pathlib import Path


def test_official_workbook_schema_and_constraints():
    train,test,budget,cap=load_organizer_workbook(Path(__file__).parents[1]/'data/bank_campaign_participant_release.xlsx')
    assert (len(train),len(test),budget,cap)==(300,100,25,22)
    assert train.y.sum()==35 and 'y' not in test
    assert not set(train.customer_id)&set(test.customer_id)
    assert 'duration' not in train and 'expected_deposit_value' in test


def test_target_yes_no_and_invalid():
    assert normalize_target(pd.Series(['yes','no'])).tolist()==[1,0]
    with pytest.raises(ValueError): normalize_target(pd.Series(['yes','maybe']))


@pytest.mark.parametrize('ids', [['a','a'],['a',None],['b','c']])
def test_metadata_join_rejects_bad_keys(ids):
    with pytest.raises(ValueError):
        join_metadata(pd.DataFrame({'customer_id':['a','b']}),pd.DataFrame({'customer_id':ids,'cost':[1,2]}))


def test_alphanumeric_ties_and_top_ten_flags():
    ids=['C'+str(i) for i in range(100)]
    r=rank_predictions(ids,np.repeat(.2,100))
    assert r.customer_id.tolist()==sorted(ids)
    assert r['rank'].tolist()==list(range(1,101))
    assert r.is_top_10.sum()==10


def test_transform_does_not_fit_unseen_categories_or_extremes():
    tr=pd.DataFrame({'number':[1.,2.,3.],'category':['a','b','a']})
    prep=preprocessor(tr); prep.fit(tr)
    means=prep.named_transformers_['num']['scale'].mean_.copy()
    categories=prep.named_transformers_['cat']['encode'].categories_[0].copy()
    prep.transform(pd.DataFrame({'number':[1e9],'category':['unknown']}))
    assert np.array_equal(means,prep.named_transformers_['num']['scale'].mean_)
    assert np.array_equal(categories,prep.named_transformers_['cat']['encode'].categories_[0])


def test_expected_net_value_can_select_lower_probability_customer():
    p=np.linspace(.1,.9,10); values=np.array([100]*6+[1]*4)
    rewards=p*values-1
    selected,info=exact_solve(np.ones(10),np.ones(10),rewards,6,6)
    assert selected==tuple(range(6))
    assert info['objective']==pytest.approx(sum(rewards[:6]))


def test_infeasible_fails():
    with pytest.raises(ValueError,match='No feasible'):
        exact_solve(np.ones(10),np.ones(10),np.ones(10),5,10)


def test_penalty_ground_state_is_feasible_net_value_optimum():
    rng=np.random.default_rng(9)
    c=rng.integers(1,5,10).astype(float); a=rng.integers(1,5,10).astype(float)
    r=rng.normal(2,10,10)
    indices,info=exact_solve(c,a,r,18,18)
    energy=energy_table(c,a,r,18,18)
    mask=int(np.argmin(energy))
    selected=[i for i in range(10) if mask&(1<<i)]
    assert len(selected)==6 and c[selected].sum()<=18 and a[selected].sum()<=18
    assert r[selected].sum()==pytest.approx(info['objective'])
    feasible=[sum(1<<i for i in s) for s in combinations(range(10),6) if c[list(s)].sum()<=18 and a[list(s)].sum()<=18]
    assert all(energy[m]==pytest.approx(-sum(r[i] for i in range(10) if m&(1<<i))/np.mean(abs(r))) for m in feasible)


def test_quadratic_slack_and_ising_match_independent_squared_formula():
    c=np.arange(10)%4+1; a=np.arange(10)%3+1; r=np.linspace(-3,4,10)
    q=build_qubo(c,a,r,25,22)
    assert q['width']==20
    for bits in np.random.default_rng(15).integers(0,2,size=(100,20)):
        expected=-r@bits[:10]+q['penalty']*((bits[:10].sum()-6)**2+(c@bits[:10]+np.dot(q['budget_slack_weights'],bits[10:15])-25)**2+(a@bits[:10]+np.dot(q['capacity_slack_weights'],bits[15:])-22)**2)
        assert qubo_energy(q,bits)==pytest.approx(expected,abs=1e-8)
        assert ising_energy(q,bits)==pytest.approx(expected,abs=1e-8)


def test_six_qubit_angles_bit_order_and_counts():
    from qiskit.quantum_info import Statevector
    p=[1,0,1,0,0,1]
    q=conversion_circuit(p,measure=False)
    assert q.num_qubits==6 and len(q.data)==6
    assert Statevector.from_instruction(q).probabilities()[37]==pytest.approx(1)
    out=simulate(list('abcdef'),p,shots=256,mc_iterations=1000,readout_error=0,engine='qiskit')
    assert out['ideal_measurement_counts']=={'100101':256}
    assert out['quantum_estimated_probability']==1
    assert out['quantum_noisy_probability']==out['quantum_mitigated_probability']==1


def test_analytical_probability_against_64_outcomes():
    p=np.array([.13,.27,.38,.42,.69,.85])
    reference=sum(np.prod(np.where(bits,p,1-p)) for bits in product((0,1),repeat=6) if sum(bits)>=3)
    assert exact_tail_probability(p)==pytest.approx(reference,abs=1e-14)
    out=simulate(list('abcdef'),p,shots=2000,mc_iterations=100000,engine='numpy')
    assert abs(out['classical_monte_carlo_probability']-reference)<.008


def test_xy_measurement_preserves_weight_six():
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    qc=xy_qiskit_circuit([.2,.6],np.linspace(-3,7,1024),10,6)
    backend=AerSimulator()
    counts=backend.run(transpile(qc,backend,seed_transpiler=21),shots=512,seed_simulator=21).result().get_counts()
    assert sum(counts.values())==512
    assert all(bits.count('1')==6 for bits in counts)


def test_model_selection_independent_of_test_features():
    from quantum_campaign.data import generate_demo,validate_inputs
    from quantum_campaign.stage1 import train_and_predict
    train,test=generate_demo(seed=42,n_train=300)
    features=validate_inputs(train,test,id_col='customer_id',target='subscribed',cost_col='contact_cost',capacity_col='capacity_units',value_col='expected_value')
    a,da=train_and_predict(train,test,features,engine='numpy',qml_backend='native')
    changed=test.copy(); changed['age']=100000; changed['region']='unseen'
    b,db=train_and_predict(train,changed,features,engine='numpy',qml_backend='native')
    assert da['metrics']==db['metrics'] and da['quantum_weight']==db['quantum_weight']
    assert da['classical_chosen']==db['classical_chosen']
    assert da['training_fit_samples']==300
    assert da['out_of_fold_probabilities']==db['out_of_fold_probabilities']
