"""Stage 3 6-qubit Bernoulli amplitude encoding, noise and readout mitigation."""
from __future__ import annotations
import numpy as np


def exact_tail_probability(probs, threshold=3):
    """Poisson-binomial dynamic programming, independent-conversion assumption."""
    coeffs = np.array([1.])
    for p in probs:
        coeffs = np.convolve(coeffs, [1-float(p), float(p)])
    return float(coeffs[threshold:].sum())


def probability_sensitivities(probs, threshold=3):
    """dP(K>=threshold)/dp_i = P(K_without_i==threshold-1)."""
    return [
        float(_exact_pmf_remove_one(probs, i, threshold-1)) for i in range(len(probs))]


def _exact_pmf_remove_one(probs, ignore, target):
    out = np.array([1.])
    for i,p in enumerate(probs):
        if i != ignore:
            out = np.convolve(out, [1-float(p),float(p)])
    return out[target]


def conversion_circuit(probs, measure=True):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(len(probs),len(probs) if measure else 0)
    for i,p in enumerate(probs):
        qc.ry(float(2*np.arcsin(np.sqrt(p))), i)
    if measure:
        qc.measure(range(len(probs)),range(len(probs)))
    return qc


def counts_to_distribution(counts, n, shots):
    arr = np.zeros(1<<n, dtype=float)
    for bitstring, frequency in counts.items():
        arr[int(str(bitstring).replace(' ',''),2)] += frequency / shots
    return arr


def readout_matrix(n, error):
    """Symmetric independent bit-flip confusion matrix: observed = M @ true."""
    single = np.array([[1-error,error],[error,1-error]])
    matrix = np.array([[1.]])
    for _ in range(n):
        matrix = np.kron(single, matrix)
    return matrix


def mitigate_readout(counts, n, shots, readout_error):
    obs = counts_to_distribution(counts,n,shots)
    corrected = np.linalg.solve(readout_matrix(n,readout_error),obs)
    # A transparent projection onto the probability simplex; can introduce bias.
    corrected = np.maximum(corrected,0)
    return corrected/corrected.sum()


def tail_from_distribution(arr, n, threshold=3):
    return float(sum(float(p) for i,p in enumerate(arr) if i.bit_count()>=threshold))


def simulate(selected_ids, probs, *, shots=8192, mc_iterations=100000,
             readout_error=.025, seed=42, engine='auto'):
    if len(selected_ids)!=6 or len(probs)!=6:
        raise ValueError('Stage 3 requires exactly six customers/probabilities')
    if len(set(selected_ids))!=6 or shots<1 or mc_iterations<1 or not 0<=readout_error<.5:
        raise ValueError('Unique IDs, positive shots/iterations and readout error in [0,.5) required')
    probs = np.asarray(probs,dtype=float)
    if not np.isfinite(probs).all() or ((probs<0)|(probs>1)).any():
        raise ValueError('All Stage 3 probabilities must be in [0,1]')
    theta = 2*np.arcsin(np.sqrt(probs))
    exact = exact_tail_probability(probs)
    rng_mc = np.random.default_rng(seed+501)
    draws = rng_mc.random((mc_iterations,6)) < probs
    monte_carlo = float(np.mean(draws.sum(axis=1)>=3))
    if engine == 'auto':
        try:
            import qiskit_aer  # noqa
            engine = 'qiskit'
        except ImportError:
            engine = 'numpy'
    if engine=='qiskit':
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, ReadoutError
        q = conversion_circuit(probs)
        ideal_backend = AerSimulator(seed_simulator=seed+1)
        ideal_circuit = transpile(q,ideal_backend,optimization_level=2, seed_transpiler=seed)
        counts = ideal_backend.run(ideal_circuit,shots=shots,seed_simulator=seed+1).result().get_counts()
        noise_model = NoiseModel()
        flip = ReadoutError([[1-readout_error,readout_error],[readout_error,1-readout_error]])
        noise_model.add_all_qubit_readout_error(flip)
        noisy_backend = AerSimulator(noise_model=noise_model, seed_simulator=seed+2)
        noisy_circuit = transpile(q,noisy_backend,optimization_level=2,seed_transpiler=seed)
        noisy_counts = noisy_backend.run(noisy_circuit, shots=shots,seed_simulator=seed+2).result().get_counts()
        circuit_details = dict(backend='Qiskit AerSimulator (ideal + synthetic independent readout noise)',
                               logical_depth=q.depth(),
                               logical_qubits=6, physical_qubits=ideal_circuit.num_qubits,
                               transpiled_depth=ideal_circuit.depth(),
                               transpiled_gate_counts=dict(ideal_circuit.count_ops()),
                               transpiler_seed=seed)
        from .circuit_metrics import basis_metrics
        circuit_details.update(basis_metrics(q, seed))
    elif engine=='numpy':
        rng = np.random.default_rng(seed+1)
        ideal_samples = rng.random((shots,6))<probs
        rng_noise = np.random.default_rng(seed+2)
        noisy_samples = ideal_samples ^ (rng_noise.random((shots,6))<readout_error)
        def bit_counts(samples):
            masks = samples.astype(int) @ (1 << np.arange(6))
            unique,count = np.unique(masks,return_counts=True)
            return {format(int(mask),'06b'):int(c) for mask,c in zip(unique,count)}
        counts, noisy_counts = bit_counts(ideal_samples),bit_counts(noisy_samples)
        circuit_details = dict(backend='NumPy Bernoulli sampling reference (NOT Qiskit)',
                               logical_qubits=6, physical_qubits=None,
                               transpiled_depth=None, transpiled_gate_counts=None)
    else:
        raise ValueError('Choose engine auto, qiskit or numpy')
    ideal_est = tail_from_distribution(counts_to_distribution(counts,6,shots),6)
    noisy_est = tail_from_distribution(counts_to_distribution(noisy_counts,6,shots),6)
    corrected_distribution = mitigate_readout(noisy_counts,6,shots,readout_error)
    mitigated = tail_from_distribution(corrected_distribution,6)
    successes=sum(int(v) for k,v in counts.items() if k.count('1')>=3)
    z=1.959963984540054
    center=(ideal_est+z*z/(2*shots))/(1+z*z/shots)
    half=z*np.sqrt(ideal_est*(1-ideal_est)/shots+z*z/(4*shots*shots))/(1+z*z/shots)
    return dict(selected_customer_ids=list(selected_ids),
                customer_to_qubit_mapping=[dict(customer_id=str(id),qubit=i,probability=float(p))
                                           for i,(id,p) in enumerate(zip(selected_ids,probs))],
                probabilities=[float(p) for p in probs],
                ry_angles_radians=[float(t) for t in theta],
                threshold=3, shots=shots, monte_carlo_iterations=mc_iterations,
                successful_outcome_counts=successes,
                noisy_successful_outcome_counts=sum(int(v) for k,v in noisy_counts.items() if k.count('1')>=3),
                shot_wilson_95_interval=[float(center-half),float(center+half)],
                uncertainty_scope='Shot interval conditional on frozen model probabilities; excludes prediction/model uncertainty',
                monte_carlo_seed=seed+501, ideal_seed=seed+1, noisy_seed=seed+2,
                bit_order='Bitstrings are q5...q0; rightmost bit is customer_to_qubit_mapping[0]',
                ideal_measurement_counts=counts, noisy_measurement_counts=noisy_counts,
                quantum_estimated_probability=ideal_est,
                quantum_noisy_probability=noisy_est,
                quantum_mitigated_probability=mitigated,
                exact_analytical_probability=exact,
                classical_monte_carlo_probability=monte_carlo,
                ideal_absolute_error=abs(ideal_est-exact),
                noisy_absolute_error=abs(noisy_est-exact),
                mitigated_absolute_error=abs(mitigated-exact),
                monte_carlo_absolute_error=abs(monte_carlo-exact),
                shot_standard_error_estimate=float(np.sqrt(exact*(1-exact)/shots)),
                probability_sensitivities=probability_sensitivities(probs),
                readout_flip_probability=readout_error,
                mitigation_method='Inverse known independent readout confusion matrix + nonnegative renormalization',
                independence_assumption=True,
                noise_scope='Synthetic READOUT error only; not a full device/noisy-gate model',
                quantum_engine=engine,
                circuit=circuit_details)
