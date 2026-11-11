# Final experiment report

Official organizer data, executed 11 November 2026.

## Stage 1: validation, calibration and final fit

Numeric fields use median imputation and standard scaling. Categorical fields use most-frequent imputation and one-hot encoding with unknown-category handling. All transforms are fitted within the outer training fold. Three-fold shuffled stratification uses seed 42. No test feature values or labels participate in selection.

The quantum branch selects four encoded features with training-label ANOVA and scales them to [0,1] with clipping. A stratified 64-row subset from each training partition bounds the quadratic kernel workload. The final preprocessor/classical estimators fit all 300 training rows; the final quantum classifier uses a disclosed 64-row subset of those training rows. Row IDs and fold indices are saved in `stage1_validation.json`. No 25% holdout is discarded from the final preprocessing fit.

Feature map: RY(pi*x_i), RZ(pi*x_i), followed by nearest-neighbor RZZ(pi*x_i*x_j). Fidelity is computed with `FidelityQuantumKernel` / `ComputeUncompute` / seeded `StatevectorSampler`, 1,024 shots per circuit. `QSVC(C=2, probability=True)` uses internal training-only sigmoid fitting. The feature transforms are reused inside the inner SVC calibration, so this is not a fully nested calibration pipeline; outer-fold validation remains untouched. Calibration-bin observations are saved and plotted.

| Model | Average precision (PR-AUC) | ROC-AUC | Brier | Log loss | Precision@10 | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.134914 | 0.525714 | 0.107262 | 0.379101 | 0.100000 | 0.028571 |
| hist_gradient_boosting | 0.154945 | 0.574124 | 0.116007 | 0.426411 | 0.100000 | 0.028571 |
| quantum_kernel_svc | 0.162837 | 0.549057 | 0.100029 | 0.350823 | 0.300000 | 0.085714 |


These are pooled out-of-fold training metrics. Average precision is the reported PR-AUC convention. Precision@10 ranks the pooled 300 out-of-fold predictions, not the hidden test set. Selection uses lowest OOF Brier, then average precision, over classical models and quantum weights 0, 0.25, 0.5, 0.75, 1. The chosen weight is **1.0**: final QSVC. These selection metrics are not an independent unbiased estimate after selection. No post-selection hidden-test evaluation was performed.

A training-prevalence constant reference has Brier `(35/300)*(265/300) = 0.103056`; QSVC's OOF Brier 0.100029 is only a modest improvement. ROC-AUC 0.549 suggests weak discrimination. The final four selected features are categorical indicators; **86 of 100 test customers share the highest predicted probability**, 13 share the next probability, and one has the lowest. The official alphanumeric tie rule therefore determines the top ten. We do not add artificial jitter or alter the ranking after inspecting economics.

Top ten: **C0005, C0007, C0009, C0012, C0020, C0022, C0025, C0028, C0036, C0039**.

## Stage 2: objective, exact certificate and Hamiltonian

Let r_i = p_i*V_i-c_i. Maximize `sum(r_i*x_i)` with binary x, `sum(x)=6`, `sum(c*x)<=25`, `sum(a*x)<=22`. Every candidate is one qubit; q0 is candidate index 0, and the rightmost measured bit. `stage2_all_210_subsets.csv` records every six-customer choice, cost, capacity, objective and feasibility. Deterministic equal-objective ties retain the first combination in Stage 1 rank order.

All **210** subsets in this particular top-ten pool are feasible. This is an observed property of these candidates, not a general guarantee from the XY mixer.

The executed 10-qubit cost phase uses a **diagonal pseudo-Boolean lookup oracle, not a quadratic QUBO**:

```text
E(x) = -sum(r_i*x_i)/R + lambda * v(x)
v(x) = (sum(x)-6)^2
     + [max(sum(c_i*x_i)-25,0)/c_unit]^2
     + [max(sum(a_i*x_i)-22,0)/a_unit]^2
R = max(mean(abs(r)), 0.0001)
lambda = (sum(abs(r))/R + 1) / min_positive(v)
```

The finite 1,024-state domain permits an exhaustive minimum-positive-violation calculation. Every infeasible basis state pays more than the entire possible normalized reward range, so a feasible ground state minimizes energy at maximal net value. Feasibility is still checked independently on measured bitstrings. Tests verify objective sign and the exact optimum.

For a genuinely quadratic alternative, `quantum_campaign/qubo.py` exports coefficients to `qubo_coefficients.json`:

```text
E_Q = -r.x + L[(sum(x)-6)^2+(c.x+s_B-25)^2+(a.x+s_A-22)^2]
L = sum(abs(r)) + 1
s_B weights = [1,2,4,8,10]
s_A weights = [1,2,4,8,7]
```

This requires **20 binary variables** (10 customer + 10 slack). Integral organizer units require no discretization. Fractional inputs are rejected by this alternative rather than silently rounded. Expanding a squared term gives linear coefficient `L(v_i^2-2*b*v_i)` and pair coefficient `2*L*v_i*v_j`. With x=(1-Z)/2, `h_i=-linear_i/2-sum_j(pair_ij)/4`, `J_ij=pair_ij/4`, with the recorded offset. Random assignments independently verify direct squared, QUBO and Ising energies agree. This alternative is algebraically validated, **not the circuit executed for the official result**.

XY-QAOA initializes a uniform Dicke-type superposition of 210 weight-six states and applies an ordered ring of `RXX(beta); RYY(beta)` pairs, preserving excitation count ideally. The standard X baseline starts in |+>^10 and applies RX(2*beta). Both use identical diagonal cost phases. The main run uses depth 2, differential evolution maxiter 12, popsize 5, seed 42, no polish, expectation objective (CVaR alpha 1). Optional CVaR support is retained but no advantage is claimed without an experiment.

Main XY measurement: 8,192 shots, feasible fraction **1.0**, measured optimum gap **0.0**. Classical optimum and final hybrid objective are **6.137511924961**. The **final decision policy is exact classical certification/refinement**, even when QAOA also measures the optimum. Measured quantum and final gaps are separate fields.

Repeated matched runs use seeds 11/22/33, depths 1/2, maxiter 6, popsize 5 and 2,048 Aer shots each. Twelve trials retain full counts and circuits in `qaoa_repeated_trials.json`.

| Mixer | Depth | Mean feasible fraction | Mean measured gap | Mean optimum fraction |
|---|---:|---:|---:|---:|
| x | 1 | 0.316243 | 0.605032 | 0.006836 |
| x | 2 | 0.260091 | 0.000000 | 0.029460 |
| xy | 1 | 1.000000 | 0.000000 | 0.028809 |
| xy | 2 | 1.000000 | 0.000000 | 0.017578 |


Small-problem simulation does not establish speedup. All six XY trials measured the exact optimum; X depth 1 did not always do so. Generic state preparation and diagonal synthesis are expensive.

## Circuit metrics and backends

| Circuit | Logical qubits | Logical depth | u/cx depth | 1Q gates | 2Q gates | SWAP | Shots |
|---|---:|---:|---:|---:|---:|---:|---:|
| Feature map (one embedding) | 4 | 5 | 11 | 11 | 6 | 0 | 1,024 per fidelity circuit |
| Main XY-QAOA | 10 | 44 | 6,144 | 3,200 | 3,137 | 0 | 8,192 |
| Campaign RY circuit | 6 | 2 | 2 | 6 | 0 | 0 | 32,768 ideal + 32,768 noisy |

A fidelity experiment composes two embeddings and their inverse; the single-map row is not a claim about total kernel training gates. Aer-native QAOA depth 44 retains `initialize` and `diagonal` operations. The u/cx decomposition exposes the much larger gate cost. Physical simulator registers have the same counts as logical registers, identity mapping, no hardware coupling-map routing. No hardware physical placement is claimed. Qiskit 2.6.2, Aer 0.17.2, Qiskit Machine Learning 0.9.1.

## Stage 3: exact probability, measurements and noise

Each selected customer has frozen probability **0.1164423994227783**, and angle **0.6964642394446809 radians**. For each q_i prepare `RY(2*asin(sqrt(p_i)))|0>`. Measured one means subscription. Strings are `q5...q0`; `100101` denotes q5,q2,q0 equal to one. Success requires popcount >=3, independent of display order. Customer-to-qubit mapping is explicitly saved.

The analytical benchmark forms the generating polynomial `product((1-p_i)+p_i*z)` by dynamic programming and sums coefficients 3 through 6. The verifier independently enumerates all 64 outcomes. Monte Carlo uses 100,000 independent six-customer trials and seed 543.

| Method | P(at least 3) | Absolute error |
|---|---:|---:|
| Ideal Aer | 0.022796630859 | 0.001252666600 |
| Noisy Aer | 0.034454345703 | 0.010405048244 |
| Readout mitigation | 0.022419917201 | 0.001629380258 |
| Monte Carlo | 0.023650000000 | 0.000399297459 |


Exact reference: **0.024049297459310**. Ideal successful shots: **747**. Wilson 95% shot interval: **[0.0212356590, 0.0244694766]**, conditional on the six model probabilities; it excludes prediction uncertainty. Ideal seed 43, noise seed 44, transpiler seed 42. Readout noise is a synthetic independent symmetric flip rate 0.025 on each qubit, not device-calibrated noise. No depolarizing gate model is claimed. Mitigation inverts the known 64x64 Kronecker confusion matrix, clips negative entries and renormalizes. This is an idealized known-noise correction, not learned hardware calibration, and clipping can bias estimates. Mitigation improved the noisy error in this run but remains less accurate than ideal shots.

## Error propagation and business interpretation

Analytical derivatives satisfy `dP(K>=3)/dp_i=P(K_without_i=2)`, here 0.0935248227 per selected customer. `sensitivity_analysis.json` applies 200 independent N(0,0.01^2) perturbations, clipped to [0,1], reranks all 100 and exactly reselects six. Mean overlap with the final six is **0.67**, and resulting analytical tails range **0.030790 to 0.040636**. This is an illustrative sensitivity experiment, not calibrated uncertainty or a bootstrap confidence interval. Many original ties explain the selection instability.

The recommended six maximize net value under the mandated model and constraints, not the probability of three conversions. Expected conversions are 0.698654; modeled probability of achieving three is only about 2.4%. A bank should not interpret a positive expected net value as a high chance of three conversions. Weak predictive discrimination, coarse encoding and sensitivity warrant better training data and richer validated features before any real deployment. This project is a hackathon experiment.

## Tests and independent verification

`python -m pytest -q`: **44 passed, 0 failed, 0 skipped** in 13.17 seconds on the final recorded suite. Earlier failures were corrected: demo constraints after refit and transient Aer DLL loading during package installation. `python -m pip check` reported no broken requirements. `verify_submission.py` returned PASS on official data.

Tests cover input joins/labels, train-test independence of model selection, unchanged preprocessing under test extremes/unseen categories, alphanumeric ties, 210-subset exactness, infeasible constraints, net-value selection, energy signs, QUBO/Ising mapping, statevector equivalence, actual XY shot cardinality, six-qubit angles/bit order, noise/mitigation, analytical enumeration, Monte Carlo, and deliberate output tampering. The verifier reconstructs formulas, counts, gaps and budgets independently and compares metadata to the downloaded workbook. Hashes provide integrity checks, not cryptographic proof of scientific provenance.
