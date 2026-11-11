"""Independent correctness checks for the repository-inspired upgrades."""
import numpy as np
import pytest
from quantum_campaign.advanced_qaoa import (
    initial_dicke_state, xy_pairs, apply_xy_pair, expected_or_cvar_energy,
    xy_state_probabilities, xy_qiskit_circuit, outcome_metrics,
)
from quantum_campaign.stage2 import energy_table
from quantum_campaign.qml_qiskit_ml import parameterized_feature_map
from quantum_campaign.feature_map import numpy_state


def test_dicke_state_has_exact_hamming_weight_and_uniform_amplitudes():
    state=initial_dicke_state(10,6)
    nz=np.flatnonzero(abs(state)>1e-9)
    assert len(nz)==210
    assert all(int(i).bit_count()==6 for i in nz)
    assert np.isclose(np.linalg.norm(state),1.0)
    assert np.allclose(np.abs(state[nz])**2,1/210)


def test_xy_conserves_cardinality_with_random_angles():
    E=np.random.default_rng(10).normal(size=1<<10)
    angles=[.3,.6,.9,.2]
    prob=xy_state_probabilities(angles,E,10,6)
    assert np.isclose(np.sum(prob),1)
    assert np.isclose(sum(prob[i] for i in range(1<<10) if i.bit_count()!=6),0,atol=1e-10)
    assert sum(prob[i] for i in range(1<<10) if i.bit_count()==6)>0.999999999


def test_cvar_matches_hand_checked_lower_tail_and_mean():
    p=np.array([.2,.3,.5])
    e=np.array([0.,1.,2.])
    assert np.isclose(expected_or_cvar_energy(p,e,1.),1.3)
    assert np.isclose(expected_or_cvar_energy(p,e,.4),.5)
    with pytest.raises(ValueError):
        expected_or_cvar_energy(p,e,0.)


def test_xy_pair_acts_unitarily_and_preserves_excitation():
    vec=np.zeros(4,dtype=complex);vec[1]=1
    apply_xy_pair(vec,0,1,.4)
    assert np.isclose(sum(abs(vec)**2),1.)
    assert np.isclose(vec[0],0) and np.isclose(vec[3],0)
    assert np.allclose([vec[1],vec[2]],[np.cos(.4),-1j*np.sin(.4)])


def test_outcome_metrics_feasible_probability_never_greater_than_six_mass():
    E=energy_table(np.ones(10),np.ones(10),np.ones(10),6,6)
    prob=xy_state_probabilities([.1,.4],E,10,6)
    diag=outcome_metrics(prob,np.ones(10),np.ones(10),np.ones(10),6,6,63)
    assert diag['feasible_state_count']==210
    assert abs(diag['feasible_probability']-1.)<1e-9
    assert diag['cardinality_probability']>=diag['feasible_probability']-1e-9


def test_xy_qiskit_state_equivalence_if_installed():
    pytest.importorskip('qiskit')
    from qiskit.quantum_info import Statevector
    E=np.random.default_rng(11).normal(size=2**4)
    angles=[.25,.42]
    qc=xy_qiskit_circuit(angles,E,4,2,measure=False)
    actual=Statevector.from_instruction(qc).probabilities()
    reference=xy_state_probabilities(angles,E,4,2)
    assert np.allclose(actual,reference,atol=1e-9)


def test_parameterized_qiskit_ml_map_equals_reference_if_installed():
    pytest.importorskip('qiskit')
    from qiskit.quantum_info import Statevector
    params=parameterized_feature_map(4)
    values=np.array([.13,.23,.31,.89])
    circuit=params.assign_parameters(values)
    actual=Statevector.from_instruction(circuit).data
    assert np.allclose(actual,numpy_state(values),atol=1e-9)


def test_qiskit_machine_learning_api_if_installed():
    pytest.importorskip('qiskit_machine_learning')
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.algorithms import QSVC
    kernel=FidelityQuantumKernel(feature_map=parameterized_feature_map(2),
                                 max_circuits_per_job=16)
    samples=np.array([[0.,0.],[.2,.1],[.8,.8],[.9,.7]])
    labels=np.array([0,0,1,1])
    qsvc=QSVC(quantum_kernel=kernel,probability=False)
    qsvc.fit(samples,labels)
    assert qsvc.predict(samples).shape==(4,)
