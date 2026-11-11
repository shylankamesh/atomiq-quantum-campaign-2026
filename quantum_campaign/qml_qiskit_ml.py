"""Optional Qiskit Machine Learning integration for probability estimation.

Uses the published API pattern from stfc/qiskit-machine-learning:
FidelityQuantumKernel + QSVC. No upstream implementation is vendored.
Large kernels are deliberately capped in Stage 1 (O(N^2) circuit pairs).
"""
from __future__ import annotations
import numpy as np


def parameterized_feature_map(n: int = 4):
    """Exactly match quantum_campaign.feature_map.feature_circuit(x)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    x=ParameterVector('x',n)
    qc=QuantumCircuit(n)
    for i in range(n):
        qc.ry(np.pi*x[i],i)
        qc.rz(np.pi*x[i],i)
    for i in range(n-1):
        qc.rzz(np.pi*x[i]*x[i+1],i,i+1)
    return qc


def make_qsvc(n=4, *, seed=42, max_circuits_per_job=128):
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit_machine_learning.state_fidelities import ComputeUncompute
    from qiskit.primitives import StatevectorSampler
    fidelity = ComputeUncompute(sampler=StatevectorSampler(default_shots=1024, seed=seed))
    kernel = FidelityQuantumKernel(feature_map=parameterized_feature_map(n), fidelity=fidelity,
                                   enforce_psd=True, max_circuits_per_job=max_circuits_per_job)
    return QSVC(quantum_kernel=kernel, C=2., probability=True, random_state=seed)


def fit_predict_qsvc(train_features, train_labels, val_features, test_features, *,
                     seed: int = 42, max_circuits_per_job: int = 128):
    """Fit kernel QSVC with internal Platt calibration and predict P(y=1).

    QSVC's probability=True uses SVC's internal cross-validation calibration;
    test labels are never provided. Uses exact/reference fidelity unless the
    installed Qiskit Machine Learning sampler is configured otherwise.
    """
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.algorithms import QSVC
    np.random.seed(seed)  # Qiskit/Scikit subsystems can depend on global RNG.
    feature_map = parameterized_feature_map(train_features.shape[1])
    kernel = FidelityQuantumKernel(feature_map=feature_map,
                                   enforce_psd=True,
                                   max_circuits_per_job=max_circuits_per_job)
    model = make_qsvc(train_features.shape[1], seed=seed, max_circuits_per_job=max_circuits_per_job)
    model.fit(train_features,np.asarray(train_labels))
    q_val = np.asarray(model.predict_proba(val_features))[:,1]
    q_test = np.asarray(model.predict_proba(test_features))[:,1]
    if not np.isfinite(q_val).all() or not np.isfinite(q_test).all():
        raise RuntimeError('QSVC produced invalid probabilities')
    return q_val, q_test
