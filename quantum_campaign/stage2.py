"""Stage 2 exact combinatorial benchmark plus a shot-based QAOA candidate generator.

This uses an exact diagonal pseudo-Boolean Hamiltonian with quadratic cardinality
and squared *positive violation* penalties for inequalities; it is not falsely
advertised as a 10-variable quadratic QUBO. See README for slack-variable QUBO.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np
from scipy.optimize import differential_evolution
from .advanced_qaoa import optimize_xy, xy_state_probabilities, xy_qiskit_circuit, outcome_metrics, expected_or_cvar_energy


def evaluate_subset(indices, costs, capacities, rewards, budget, capacity_limit, select_count=6):
    return dict(feasible=(len(indices)==select_count and
                          np.sum(costs[list(indices)]) <= budget + 1e-8 and
                          np.sum(capacities[list(indices)]) <= capacity_limit + 1e-8),
                cost=float(sum(costs[i] for i in indices)),
                capacity=float(sum(capacities[i] for i in indices)),
                objective=float(sum(rewards[i] for i in indices)))


def exact_solve(costs, capacities, rewards, budget, capacity_limit, select_count=6):
    best, best_info = None, None
    for subset in combinations(range(len(rewards)), select_count):
        info = evaluate_subset(subset, costs, capacities, rewards, budget, capacity_limit, select_count)
        if info['feasible'] and (best_info is None or info['objective'] > best_info['objective'] + 1e-12):
            best, best_info = tuple(subset), info
    if best is None:
        raise ValueError('No feasible subset of six under the supplied budget/capacity limits')
    return best, best_info


def energy_table(costs, capacities, rewards, budget, capacity_limit, select_count=6):
    """Penalty-oracle energies on all 2^10 basis states; q0=first candidate."""
    n = len(rewards)
    masks = np.arange(1 << n, dtype=int)
    bits = ((masks[:, None] >> np.arange(n)) & 1)
    counts = bits.sum(axis=1)
    spend = bits @ costs
    used = bits @ capacities
    reward = bits @ rewards
    # Normalize score so penalty coefficients are comparable across objectives.
    reward_unit = max(float(np.mean(np.abs(rewards))), 1e-4)
    budget_unit = max(float(np.min(costs[costs > 0])) if np.any(costs > 0) else 1., 1.)
    capacity_unit = max(float(np.min(capacities[capacities > 0])) if np.any(capacities > 0) else 1., 1.)
    violation = ((counts-select_count)**2
              + (np.maximum(spend-budget, 0)/budget_unit)**2
              + (np.maximum(used-capacity_limit, 0)/capacity_unit)**2)
    # Exhaustively bound the finite 10-bit domain, including fractional inputs.
    # Any positive violation costs more than the entire possible reward range.
    positive = violation[violation > 0]
    coefficient = (float(np.sum(np.abs(rewards)))/reward_unit + 1.) / float(positive.min()) if len(positive) else 1.
    losses = -reward / reward_unit + coefficient*violation
    return losses


def state_probabilities(angles: np.ndarray, energies: np.ndarray, n: int) -> np.ndarray:
    """Reference statevector evaluation of p-layer QAOA, without sampling."""
    layer_count = len(angles)//2
    gammas, betas = angles[:layer_count], angles[layer_count:]
    vec = np.full(1<<n, (1<<n)**-.5, dtype=complex)
    for gamma, beta in zip(gammas, betas):
        vec *= np.exp(-1j*gamma*energies)
        c, s = np.cos(beta), -1j*np.sin(beta)
        for q in range(n):
            idx = np.arange(1<<n)
            low = idx[(idx & (1 << q)) == 0]
            high = low | (1 << q)
            a, b = vec[low].copy(), vec[high].copy()
            vec[low] = c*a + s*b
            vec[high] = s*a + c*b
    return np.abs(vec)**2


def qaoa_circuit(angles, energies, n, measure=True):
    """Qiskit representation of the *same* diagonal-oracle QAOA circuit.

    Generic diagonal gates can be expensive to transpile; count 2Q gates after
    transpilation, never characterize this as hardware-efficient by default.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import DiagonalGate
    layer_count = len(angles)//2
    gammas, betas = angles[:layer_count], angles[layer_count:]
    qc = QuantumCircuit(n, n if measure else 0)
    qc.h(range(n))
    for gamma, beta in zip(gammas, betas):
        qc.append(DiagonalGate(np.exp(-1j*gamma*energies).tolist()), range(n))
        for i in range(n):
            qc.rx(float(2*beta), i)
    if measure:
        qc.measure(range(n), range(n))
    return qc


def optimize_qaoa(energies, n, *, seed=42, layers=2, maxiter=12, cvar_alpha=1.0):
    """Minimize expected penalized energy using a fixed-seed classical outer loop."""
    bounds = [(0,2*np.pi)]*layers + [(0,np.pi)]*layers
    def objective(params):
        return expected_or_cvar_energy(state_probabilities(params, energies, n), energies, cvar_alpha)
    result = differential_evolution(objective, bounds, maxiter=maxiter, popsize=5,
                                    polish=False, tol=.03, rng=np.random.default_rng(seed),
                                    updating='immediate')
    return np.asarray(result.x), float(result.fun)


def solve(top10, *, budget, capacity_limit, cost_col, capacity_col, value_col,
          objective='expected_profit', seed=42, shots=4096, engine='auto',
          qaoa_layers=2, qaoa_iterations=12, qaoa_mixer='xy', compare_mixers=False,
          qaoa_cvar=1.0):
    """top10 is already ordered by Stage-1 rank, with 10 distinct IDs."""
    if len(top10) != 10:
        raise ValueError(f'Stage 2 requires 10 candidates; received {len(top10)}')
    ids = top10['customer_id'].astype(str).tolist()
    if len(set(ids)) != 10:
        raise ValueError('Stage 2 requires unique customer IDs')
    costs = top10[cost_col].to_numpy(dtype=float)
    capacities = top10[capacity_col].to_numpy(dtype=float)
    probs = top10['predicted_probability'].to_numpy(dtype=float)
    if not all(np.isfinite(v).all() for v in (costs, capacities, probs)) or np.any(costs<0) or np.any(capacities<0) or np.any((probs<0)|(probs>1)):
        raise ValueError('Stage 2 inputs must be finite, nonnegative and probabilities bounded')
    if objective == 'expected_conversions':
        rewards = probs
    elif objective == 'expected_revenue':
        if not value_col:
            raise ValueError('expected_revenue requires --value-column')
        rewards = probs*top10[value_col].to_numpy(dtype=float)
    elif objective == 'expected_profit':
        if not value_col:
            raise ValueError('expected_profit requires --value-column')
        rewards = probs*top10[value_col].to_numpy(dtype=float)-costs
    else:
        raise ValueError(f'Unknown objective {objective!r}')

    exact_indices, exact_info = exact_solve(costs, capacities, rewards, budget, capacity_limit)
    energies = energy_table(costs, capacities, rewards, budget, capacity_limit)
    if not 0 < qaoa_cvar <= 1:
        raise ValueError('qaoa_cvar must be in (0, 1]')
    if qaoa_mixer not in ('xy','x'):
        raise ValueError('qaoa_mixer must be xy or x')
    if qaoa_mixer == 'xy':
        angles, expectation, optimization_history = optimize_xy(
            energies, 10, 6, seed=seed, layers=qaoa_layers, maxiter=qaoa_iterations,
            cvar_alpha=qaoa_cvar)
        probs_qaoa = xy_state_probabilities(angles, energies, 10, 6)
    else:
        angles, expectation = optimize_qaoa(energies, 10, seed=seed,
                                            layers=qaoa_layers, maxiter=qaoa_iterations, cvar_alpha=qaoa_cvar)
        probs_qaoa = state_probabilities(angles, energies, 10)
        optimization_history = []
    optimum_mask = sum(1 << i for i in exact_indices)
    theoretical_metrics = outcome_metrics(probs_qaoa,costs,capacities,rewards,
                                          budget,capacity_limit,optimum_mask)
    mixer_comparison = None
    if compare_mixers:
        other = 'x' if qaoa_mixer == 'xy' else 'xy'
        if other == 'xy':
            other_angles, other_expectation, _ = optimize_xy(
                energies,10,6,seed=seed,layers=qaoa_layers,maxiter=qaoa_iterations,
                cvar_alpha=qaoa_cvar)
            other_probs=xy_state_probabilities(other_angles,energies,10,6)
        else:
            other_angles,other_expectation=optimize_qaoa(
                energies,10,seed=seed,layers=qaoa_layers,maxiter=qaoa_iterations,
                cvar_alpha=qaoa_cvar)
            other_probs=state_probabilities(other_angles,energies,10)
        mixer_comparison = {
            qaoa_mixer: dict(energy=float(expectation),
                             **theoretical_metrics),
            other: dict(energy=float(other_expectation),
                        **outcome_metrics(other_probs,costs,capacities,rewards,
                                          budget,capacity_limit,optimum_mask)),
            'caveat':'Reference statevector, same optimizer budget; not hardware speedup or guaranteed XY superiority',
        }
    if engine == 'auto':
        try:
            import qiskit_aer  # noqa
            engine = 'qiskit'
        except ImportError:
            engine = 'numpy'
    circuit_details = dict(logical_qubits=10, qaoa_layers=qaoa_layers,
                           cvar_lower_tail_alpha=qaoa_cvar,
                           cost_phase_gate='exact pseudo-Boolean DiagonalGate',
                           mixer=qaoa_mixer,
                           initial_state=('weight-6 uniform Dicke-like state' if qaoa_mixer=='xy' else 'uniform plus'),
                           gate_depth_caveat='Generic DiagonalGate and arbitrary Dicke-state initialization may be expensive',
                           classical_outer_optimizer='scipy.differential_evolution')
    if engine == 'qiskit':
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        backend = AerSimulator(seed_simulator=seed)
        circuit = (xy_qiskit_circuit(angles, energies, 10, 6) if qaoa_mixer=='xy'
                   else qaoa_circuit(angles, energies, 10))
        compiled = transpile(circuit, backend, optimization_level=1,
                             seed_transpiler=seed)
        counts = backend.run(compiled, shots=shots, seed_simulator=seed).result().get_counts()
        circuit_details.update(backend='AerSimulator', transpiled_depth=compiled.depth(),
                               logical_depth=circuit.depth(),
                               transpiled_gate_counts=dict(compiled.count_ops()),
                               physical_qubits=compiled.num_qubits)
        from .circuit_metrics import basis_metrics
        circuit_details.update(basis_metrics(circuit, seed))
    elif engine == 'numpy':
        rng = np.random.default_rng(seed)
        masks = rng.choice(1024, p=probs_qaoa, size=shots)
        uniq, freq = np.unique(masks, return_counts=True)
        counts = {format(int(m),'010b'):int(c) for m,c in zip(uniq,freq)}
        circuit_details.update(backend='NumPy statevector reference (NOT Aer / hardware)',
                               transpiled_depth=None, transpiled_gate_counts=None,
                               physical_qubits=None)
    else:
        raise ValueError('Engine must be qiskit, numpy or auto')

    ranked_shots = []
    for string, freq in counts.items():
        subset = tuple(i for i in range(10) if int(string,2) & (1<<i))
        info = evaluate_subset(subset, costs, capacities, rewards, budget, capacity_limit)
        ranked_shots.append(dict(bitstring=string, shots=int(freq), **info))
    ranked_shots.sort(key=lambda item:(item['feasible'], item['objective'], item['shots']),reverse=True)
    measured_feasible_shots = sum(r['shots'] for r in ranked_shots if r['feasible'])
    measured_exact_six_shots = sum(r['shots'] for r in ranked_shots if r['bitstring'].count('1') == 6)
    measured_optimum_shots = sum(r['shots'] for r in ranked_shots
                                  if int(r['bitstring'],2) == optimum_mask)
    best_feasible = next((x for x in ranked_shots if x['feasible']), None)
    qaoa_objective = None if best_feasible is None else best_feasible['objective']
    qaoa_gap = None if qaoa_objective is None else max(0.,exact_info['objective']-qaoa_objective)
    # Explicit hybrid policy: classical exact certificate provides final optimal subset.
    # This makes no claim that QAOA attained the optimal objective or speedup.
    selected = list(exact_indices)
    data = dict(
        candidate_ids=ids, selected_customer_ids=[ids[i] for i in selected],
        selected_candidate_indices=selected, selected_bitstring=format(sum(1<<i for i in selected),'010b'),
        objective_type=objective, total_budget=float(budget), total_capacity=float(capacity_limit),
        objective_formula='sum(x_i * (p_i * expected_deposit_value_i - contact_cost_i))' if objective=='expected_profit' else objective,
        budget_used=exact_info['cost'], capacity_used=exact_info['capacity'],
        objective_value=exact_info['objective'], feasible=True,
        classical_bruteforce_optimum=exact_info['objective'],
        final_optimality_gap=0.,
        qaoa_best_measured_feasible=best_feasible,
        qaoa_best_measured_objective=qaoa_objective, qaoa_measured_optimality_gap=qaoa_gap,
        qaoa_found_optimum=bool(qaoa_gap is not None and qaoa_gap < 1e-8),
        most_frequent_bitstrings=sorted(ranked_shots,key=lambda x:x['shots'], reverse=True)[:10],
        qaoa_measurement_counts={k:int(v) for k,v in sorted(counts.items())},
        qaoa_angles=[float(x) for x in angles], qaoa_optimized_penalized_energy_or_cvar=expectation,
        qaoa_expected_penalized_energy=float(np.dot(probs_qaoa, energies)),
        qaoa_mixer=qaoa_mixer, qaoa_cvar_lower_tail_alpha=qaoa_cvar,
        qaoa_theoretical_distribution_metrics=theoretical_metrics,
        qaoa_mixer_comparison=mixer_comparison,
        qaoa_objective_evaluations=len(optimization_history) if optimization_history else None,
        measured_feasible_fraction=float(measured_feasible_shots/shots),
        measured_cardinality_six_fraction=float(measured_exact_six_shots/shots),
        measured_exact_optimum_fraction=float(measured_optimum_shots/shots),
        measured_feasible_standard_error=float(np.sqrt((measured_feasible_shots/shots)*
                                         (1-measured_feasible_shots/shots)/shots)),
        shots=shots, quantum_engine=engine, circuit=circuit_details,
        final_selection_policy='Exact classical certification/refinement, with QAOA measured candidate comparison',
        oracle_policy='Exact diagonal pseudo-Boolean objective with squared inequality overage; NOT 10-bit QUBO',
        oracle_penalty_policy='Coefficient=(sum(abs(reward))/reward_unit+1)/minimum_positive_normalized_violation across all 1024 states',
        oracle_ground_state_mask=int(np.argmin(energies)),
        enumeration=[dict(candidate_indices=list(subset), customer_ids=[ids[i] for i in subset],
                          **evaluate_subset(subset,costs,capacities,rewards,budget,capacity_limit))
                     for subset in combinations(range(10),6)],
        inputs=[dict(customer_id=ids[i], p=float(probs[i]), cost=float(costs[i]),
                     capacity=float(capacities[i]), reward=float(rewards[i]),
                     expected_deposit_value=float(top10.iloc[i][value_col]) if value_col else None) for i in range(10)],
    )
    return data
