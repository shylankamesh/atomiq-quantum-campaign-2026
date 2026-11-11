# Atomiq Quantum Campaign Optimizer

A reproducible hybrid quantum-classical term-deposit campaign, executed on the official Atomiq 2026 participant workbook. Team members: **Shylan Kamesh S, Sai Santhosh T, Derin Godson**.

**Verified result:** contact **C0005, C0007, C0009, C0020, C0022, C0036**. Expected net value **6.137511925**, contact cost **9/25**, capacity **19/22**. The six-qubit Aer estimate of at least three subscriptions is **0.022796630859375** (747 successes / 32,768 shots). The independent exact probability is **0.024049297459310**.

These are simulator results, not hardware execution or evidence of quantum advantage. Predictive performance is modest and the selected four-dimensional classifier creates many ties. This limitation materially affects the business recommendation.

## Quick reproduction

Tested with Python 3.10.11 on Windows x64. Run from this repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python fetch_data.py
python run_pipeline.py --workbook data/bank_campaign_participant_release.xlsx --engine qiskit --qml-backend qiskit_ml --compare-mixers --qaoa-shots 8192 --simulation-shots 32768
python verify_submission.py
python -m pytest -q
python complete_evidence.py
```

`requirements.txt` pins runtime packages exactly; `requirements-lock.txt` records the complete executed environment. Circuit plots additionally require `pylatexenc==2.11`, included in requirements. The official workbook is downloaded from the organizers and SHA-256 checked by `fetch_data.py`. It is kept out of the public Git repository because redistribution licensing is unspecified; the downloadable local submission archive includes the supplied public participant release for reproducibility. No hidden test labels are available or used.

The complete Stage 1 fit took 319.82 seconds. A final stages 2/3 refresh took 7.97 seconds and reused frozen, hash-checked Stage 1 results after adding circuit metrics. To refresh downstream stages without retraining, add `--resume-stage1` to the same command. It rejects a changed workbook, seed, quantum backend or frozen prediction hash. It never changes Stage 1 rankings to improve Stage 2 feasibility.

## Official problem and data provenance

[Official resource folder](https://drive.google.com/drive/folders/1vjIM7mD7XjUjvfotAXF4uVfE2cYTS7Sp) and [Devpost rules](https://atomiq-hackathon-qff-26-iem.devpost.com/rules) define the challenge. The four-page problem statement was visually inspected. The participant workbook README and constraints are the numerical authority.

- `Stage1_Train`: 300 rows, 35 yes and 265 no labels, 15 predictive features plus `customer_id` and `y`.
- `Stage1_Test`: 100 rows with the same feature schema and no labels. No hidden-label metrics are reported.
- `Stage2_Data`: operational metadata joined by exact `customer_id`; its extra rows are not treated as training examples.
- `Stage2_Constraints`: N=10, K=6, C=25, H=22.
- `README`: expected net value is `predicted_probability * expected_deposit_value - contact_cost`; probability ties use ascending alphanumeric customer ID.

Source workbook SHA-256: `d2f8e63705af2883c7618c463a6ac2f351119d59a8869d2677f8825148e9a709`. `run_manifest.json` records source URL, versions, environment and seeds; `validation_manifest.json` hashes mandatory results. IDs are preserved as strings. Train/test IDs must be disjoint. Duplicate/missing IDs, nonfinite or negative operational values, missing joins, labeled test data and incorrect row counts are rejected. `duration` is excluded to avoid post-contact leakage, and operational metadata is excluded from predictors. Currency units are not specified in the README, so values are reported in organizer units, not rupees or dollars.

## Pipeline and quantum/classical boundary

1. Training-only preprocessing and cross-validation compare logistic regression, histogram gradient boosting, and a genuine four-qubit Qiskit fidelity-kernel QSVC.
2. Freeze the chosen model/blend policy, refit training components, predict and rank all 100 once, and persist full-precision probabilities. The persisted CSV is reloaded for downstream stages.
3. Top ten enter ten-qubit fixed-weight XY-QAOA. Classical optimization chooses circuit parameters using exact statevector expectation, then genuine Aer executes the optimized circuit. Enumerating 210 subsets independently certifies the final six.
4. Six frozen Stage 1 probabilities prepare six independent RY qubits. Ideal and noisy Aer shot estimates are compared with classical dynamic programming and Monte Carlo.

The kernel is simulated by Qiskit `StatevectorSampler`, the QAOA parameter loop uses the independently implemented NumPy statevector, and QAOA final measurements and campaign circuits use Aer. Parameter search is therefore classical-assisted simulation; it is not a hardware-in-the-loop training claim.

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

`python -m pytest -q`: **44 passed, 0 failed, 0 skipped** in 13.17 seconds on the final recorded suite. The demo feasibility failure was corrected by changing synthetic-only limits. A later fresh-clone run exposed intermittent Windows Aer DLL loading and shared temporary-folder permissions; the final unsandboxed clean-checkout run with a new workspace temporary directory passed all 44 tests in 18.27 seconds. See docs/CLEAN_CHECKOUT_VERIFICATION.md for the failed attempts and recovery commands. `python -m pip check` reported no broken requirements. `verify_submission.py` returned PASS on official data.

Tests cover input joins/labels, train-test independence of model selection, unchanged preprocessing under test extremes/unseen categories, alphanumeric ties, 210-subset exactness, infeasible constraints, net-value selection, energy signs, QUBO/Ising mapping, statevector equivalence, actual XY shot cardinality, six-qubit angles/bit order, noise/mitigation, analytical enumeration, Monte Carlo, and deliberate output tampering. The verifier reconstructs formulas, counts, gaps and budgets independently and compares metadata to the downloaded workbook. Hashes provide integrity checks, not cryptographic proof of scientific provenance.

## Files

- Mandatory root results: `stage1_predictions.csv`, `stage2_selection.json`, `stage3_simulation.json`.
- Evidence: `stage1_validation.json`, `run_manifest.json`, `validation_manifest.json`, `stage1_circuit_metrics.json`, `stage2_all_210_subsets.csv`, `qaoa_repeated_trials.*`, `sensitivity_analysis.json`, `qubo_coefficients.json`, `figures/`.
- Audit/report: `docs/INITIAL_AUDIT.md`, `FINAL_COMPLIANCE_AUDIT.md`, `FINAL_EXPERIMENT_REPORT.md`.
- Presentation: `slides/Atomiq_Official_Pitch.pptx`, `slides/Atomiq_Official_Pitch.pdf`.
- `fitted_model.joblib` is saved locally. Do not load untrusted pickle/joblib files. It is not needed to reproduce training and is omitted from the public repository.
- Synthetic demonstrations are isolated in `example/demo_outputs/`, never root results. `--demo` uses artificial budget 100/capacity 24 and does not represent organizer economics.

## Credits and licensing

Original project retained and corrected under MIT. [Qiskit Machine Learning](https://github.com/stfc/qiskit-machine-learning) supplies installed public FidelityQuantumKernel/QSVC APIs. [OpenQuantumComputing/QAOA](https://github.com/OpenQuantumComputing/QAOA) is cited for research concepts (fixed-weight initial states, XY mixing, CVaR); no GPL source was copied or imported. Qiskit/Aer and other dependencies retain their own licenses. The organizer workbook is a separate public resource and is not relicensed by this repository's MIT license.

## Submission

Public source: https://github.com/shylankamesh/atomiq-quantum-campaign-2026

Public view-only deck: https://github.com/shylankamesh/atomiq-quantum-campaign-2026/blob/main/slides/Atomiq_Official_Pitch.pdf

`DEVPOST_SUBMISSION.md` provides computational fields. Registered emails and Discord handles are kept in the separate local `DEVPOST_PRIVATE_DETAILS.md`, not published to GitHub. Eligibility/prior registration and the final Devpost submission must be confirmed by the team. The supplied official rules require a deck; they do not list a mandatory video. The legacy demo script is optional supplementary material.
