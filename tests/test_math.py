import numpy as np
from itertools import combinations
from quantum_campaign.feature_map import numpy_state,fidelity_kernel
from quantum_campaign.stage2 import exact_solve,energy_table,state_probabilities
from quantum_campaign.stage3 import exact_tail_probability,probability_sensitivities,mitigate_readout,tail_from_distribution


def test_feature_normalization_kernel():
    state=numpy_state([.1,.35,.6,.8])
    assert np.isclose(np.vdot(state,state),1)
    assert np.isclose(fidelity_kernel(state.reshape(1,-1),state.reshape(1,-1))[0,0],1.)


def test_exact_solver_constraint():
    costs=np.array([10,12,8,20,18,17,9,7,16,10],float)
    cap=np.array([1,2,2,3,5,2,2,1,4,1],float)
    reward=np.linspace(.1,1.,10)
    subset, info = exact_solve(costs,cap,reward,60,12)
    assert len(subset)==6 and info['feasible']
    assert all(sum(costs[list(t)])>60 or sum(cap[list(t)])>12 or sum(reward[list(t)])<=info['objective']+1e-9 for t in combinations(range(10),6))


def test_qaoa_state_probability_normalization():
    x=np.ones(10)
    energy=energy_table(x,x,x,8,8)
    distribution=state_probabilities([.2,.6,.5,.1],energy,10)
    assert len(distribution)==1024
    assert np.isclose(sum(distribution),1,atol=1e-10)
    assert min(distribution)>=0


def test_tail_exact():
    assert np.isclose(exact_tail_probability([.5]*6),42/64)
    assert exact_tail_probability([0.]*6)==0
    assert exact_tail_probability([1.]*6)==1
    assert np.allclose(probability_sensitivities([.5]*6),[10/32]*6)


def test_readout_mitigation():
    from quantum_campaign.stage3 import readout_matrix
    actual=np.zeros(64);actual[7]=1
    observed=readout_matrix(6,.03)@actual
    synthetic_counts={format(i,'06b'):float(obs*1000000) for i,obs in enumerate(observed)}
    corrected=mitigate_readout(synthetic_counts,6,1000000,.03)
    assert tail_from_distribution(corrected,6)>0.999999


def test_qiskit_featuremap_equivalence_if_installed():
    import pytest
    pytest.importorskip('qiskit')
    from qiskit.quantum_info import Statevector
    from quantum_campaign.feature_map import feature_circuit
    values=[.15,.32,.66,.81]
    q=Statevector.from_instruction(feature_circuit(values)).data
    assert np.allclose(numpy_state(values),q,atol=1e-10)


def test_qiskit_qaoa_state_equivalence_if_installed():
    import pytest
    pytest.importorskip('qiskit')
    from qiskit.quantum_info import Statevector
    from quantum_campaign.stage2 import qaoa_circuit
    energy=np.linspace(-1,2,2**3)
    angles=[.3,.7]
    state=Statevector.from_instruction(qaoa_circuit(angles,energy,3,measure=False)).data
    assert np.allclose(np.abs(state)**2,state_probabilities(angles,energy,3),atol=1e-10)
