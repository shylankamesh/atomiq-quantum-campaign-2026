"""Same 4-qubit entangled feature-map states, via Qiskit or a checked NumPy reference.

Circuit per feature x in [0,1]: RY(pi*x) then RZ(pi*x) on each qubit,
then RZZ(pi*x_i*x_(i+1)) on neighboring qubits. Little-endian ordering.
"""
from __future__ import annotations
import numpy as np


def feature_circuit(values: np.ndarray):
    from qiskit import QuantumCircuit
    values = np.asarray(values, dtype=float)
    qc = QuantumCircuit(len(values))
    for i, v in enumerate(values):
        qc.ry(float(np.pi * v), i)
        qc.rz(float(np.pi * v), i)
    for i in range(len(values)-1):
        qc.rzz(float(np.pi * values[i]*values[i+1]), i, i+1)
    return qc


def numpy_state(values: np.ndarray) -> np.ndarray:
    """Analytically evaluate this circuit (not a quantum-hardware execution)."""
    values = np.asarray(values, dtype=float)
    states = []
    for v in values:
        theta = np.pi * v
        states.append(np.array([
            np.cos(theta / 2) * np.exp(-.5j * theta),
            np.sin(theta / 2) * np.exp(.5j * theta),
        ], dtype=complex))
    out = np.array([1+0j])
    for st in states:
        out = np.kron(st, out)
    basis = np.arange(2 ** len(values))
    for i in range(len(values)-1):
        zi = 1 - 2 * ((basis >> i) & 1)
        zj = 1 - 2 * ((basis >> (i+1)) & 1)
        out *= np.exp(-.5j * np.pi * values[i] * values[i+1] * zi * zj)
    return out


def state_embeddings(x: np.ndarray, engine: str = 'auto') -> np.ndarray:
    if engine == 'auto':
        try:
            import qiskit  # noqa
            engine = 'qiskit'
        except ImportError:
            engine = 'numpy'
    if engine == 'qiskit':
        from qiskit.quantum_info import Statevector
        return np.stack([Statevector.from_instruction(feature_circuit(row)).data for row in x])
    if engine == 'numpy':
        return np.stack([numpy_state(row) for row in x])
    raise ValueError('Unsupported quantum engine; choose auto, qiskit or numpy')


def fidelity_kernel(a_states: np.ndarray, b_states: np.ndarray) -> np.ndarray:
    return np.abs(a_states.conj() @ b_states.T) ** 2
