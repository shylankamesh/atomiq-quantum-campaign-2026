"""Independently implemented fixed-cardinality XY-QAOA.

Research ideas: OpenQuantumComputing/QAOA (GPL-3.0). NO upstream code is copied
or imported. The project retains its own MIT license and uses only Qiskit gates.

Dicke initial state: uniform superposition over weight-k states, exactly
normalized; XY (XX+YY) mixer commutes with total excitation number, hence
cardinality is enforced by the ansatz rather than penalized. Budget/capacity
are enforced by an exact diagonal penalty oracle; this is NOT a 10-bit QUBO.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np
from scipy.optimize import differential_evolution


def weight_masks(n: int, k: int) -> np.ndarray:
    if not 0 <= k <= n:
        raise ValueError('0 <= k <= n required')
    return np.array([sum(1 << j for j in positions)
                     for positions in combinations(range(n), k)], dtype=int)


def initial_dicke_state(n: int, k: int) -> np.ndarray:
    masks = weight_masks(n, k)
    state = np.zeros(1 << n, dtype=complex)
    state[masks] = 1. / np.sqrt(len(masks))
    return state


def xy_pairs(n: int, topology: str = 'ring') -> list[tuple[int,int]]:
    if topology not in ('ring','chain'):
        raise ValueError('Use ring or chain')
    pairs = [(i, i+1) for i in range(n-1)]
    if topology == 'ring' and n > 2:
        pairs.append((n-1, 0))
    return pairs


def apply_xy_pair(state: np.ndarray, a: int, b: int, beta: float):
    """Apply exp[-i beta/2 (XX+YY)] for q_a,q_b, in-place.

    In subspace |01>,|10>, this is [[cos beta,-i sin beta],
    [-i sin beta,cos beta]]. Qiskit equivalent: rxx(beta); ryy(beta).
    """
    mask_a, mask_b = 1<<a, 1<<b
    indices = np.arange(len(state))
    low = indices[((indices & mask_a) == 0) & ((indices & mask_b) != 0)]
    high = low ^ mask_a ^ mask_b
    c, s = np.cos(beta), -1j*np.sin(beta)
    l, h = state[low].copy(), state[high].copy()
    state[low] = c*l + s*h
    state[high] = s*l + c*h


def xy_state_probabilities(angles, energies, n: int, k: int = 6,
                           topology: str = 'ring') -> np.ndarray:
    angles = np.asarray(angles, dtype=float)
    if len(angles) % 2 or not len(angles):
        raise ValueError('angles must contain gammas then betas, each of length p')
    p = len(angles)//2
    gammas, betas = angles[:p], angles[p:]
    state = initial_dicke_state(n,k)
    edges = xy_pairs(n,topology)
    for gamma,beta in zip(gammas,betas):
        state *= np.exp(-1j*gamma*np.asarray(energies))
        for a,b in edges:
            apply_xy_pair(state,a,b,beta)
    return np.abs(state)**2


def xy_qiskit_circuit(angles, energies, n: int, k: int = 6,
                      topology: str = 'ring', measure: bool = True):
    """Qiskit circuit matching the NumPy reference, with expensive state prep.

    initialize() is simulation oriented; do not claim shallow hardware depth.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import DiagonalGate
    angles = np.asarray(angles, dtype=float)
    p = len(angles)//2
    qc = QuantumCircuit(n, n if measure else 0)
    qc.initialize(initial_dicke_state(n,k), list(range(n)))
    for gamma,beta in zip(angles[:p],angles[p:]):
        qc.append(DiagonalGate(np.exp(-1j*gamma*energies).tolist()),range(n))
        for a,b in xy_pairs(n,topology):
            qc.rxx(float(beta),a,b)
            qc.ryy(float(beta),a,b)
    if measure:
        qc.measure(range(n),range(n))
    return qc


def expected_or_cvar_energy(probabilities, energies, alpha: float = 1.0) -> float:
    """Lower-tail Conditional Value at Risk of penalized *energy*.

    For alpha=1 this is ordinary expectation. For alpha<1 optimize the lowest
    energy alpha mass; explicitly not a quantum speedup or hardware measurement.
    """
    if not 0 < alpha <= 1:
        raise ValueError('CVaR alpha must lie in (0,1]')
    p=np.asarray(probabilities,dtype=float)
    e=np.asarray(energies,dtype=float)
    if p.shape != e.shape or np.any(p < -1e-12) or abs(p.sum()-1)>1e-8:
        raise ValueError('Invalid probability/energy distribution')
    if alpha == 1:
        return float(np.dot(p,e))
    order=np.argsort(e,kind='stable')
    lower_p=p[order]
    cumul_before=np.r_[0., np.cumsum(lower_p[:-1])]
    weights=np.maximum(0.,np.minimum(lower_p, alpha-cumul_before))
    return float(np.dot(weights,e[order])/alpha)


def optimize_xy(energies, n: int, k: int = 6, *,
                seed: int = 42, layers: int = 2, maxiter: int = 12,
                topology: str = 'ring', cvar_alpha: float = 1.0):
    bounds = [(0,2*np.pi)]*layers + [(0,np.pi)]*layers
    history=[]
    def objective(params):
        value = expected_or_cvar_energy(xy_state_probabilities(params,energies,n,k,topology),
                                        energies,cvar_alpha)
        history.append(value)
        return value
    res = differential_evolution(objective,bounds,maxiter=maxiter,popsize=5,
                                 polish=False,tol=.03,rng=np.random.default_rng(seed),
                                 updating='immediate')
    return np.asarray(res.x),float(res.fun),history


def outcome_metrics(probabilities, costs, capacities, rewards, budget, capacity_limit,
                    optimum_mask, select_count=6):
    n=len(costs)
    masks=np.arange(1<<n)
    bits=(masks[:,None]>>np.arange(n))&1
    valid=((bits.sum(axis=1)==select_count)&
           (bits@costs<=budget+1e-8)&(bits@capacities<=capacity_limit+1e-8))
    return dict(cardinality_probability=float(sum(probabilities[bits.sum(axis=1)==select_count])),
                feasible_probability=float(sum(probabilities[valid])),
                optimum_probability=float(probabilities[optimum_mask]),
                feasible_state_count=int(sum(valid)))
