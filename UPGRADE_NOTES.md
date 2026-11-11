> Historical synthetic-demo material. Superseded for official results by README.md and FINAL_EXPERIMENT_REPORT.md. Video is optional under the supplied rules.

# Atomiq v1.1 — GitHub-research integration, 11 November 2026

## Scope and provenance

- Requested reference 1: https://github.com/stfc/qiskit-machine-learning (Apache-2.0). Integration is via an optional installed `qiskit-machine-learning` package (`FidelityQuantumKernel` and `QSVC`). No source code was copied.
- Requested reference 2: https://github.com/OpenQuantumComputing/QAOA (GPL-3.0). Its architecture inspired a separate **independent** implementation of fixed-weight initial state, XY mixer, optional CVaR objective and multi-seed comparisons. No code from this GPL repository was copied or imported. The Atomiq project remains MIT licensed.
- The exact contest dataset and detailed constraints were unavailable. The results below use generated example data only.

## Main corrections to the previous archive

| Area | Old | Improved |
|---|---|---|
| Stage 1 kernel | Self-contained statevector fidelity + sklearn SVC | Added `--qml-backend qiskit_ml` to use **Qiskit ML FidelityQuantumKernel + QSVC**. Preserved `native` for quick mathematical checks. |
| Prediction evaluation | PR-AUC, Brier, top-10 metrics | Added ROC-AUC and log-loss; validation may choose quantum weight `0` when quantum classifier hurts Brier, avoiding manufactured improvements. |
| Stage 2 mixer | 10-qubit unconstrained X mixer | Default **XY ring mixer**, initialized in a uniform weight-six Dicke-like state. Exactly-six cardinality preserved in ideal simulation. |
| Budget/capacity | Penalized oracle, post-selection feasibility | Same exact-inequality oracle and feasibility check; **XY does not itself preserve budget or capacity**. |
| Classical check | Exhaustive 210 subset optimum | Retained as an exact selection certificate. No unjustified quantum speedup claim. |
| Optimization diagnostics | QAOA measured best subset and gap | Added all measurement counts, exact-six mass, feasible mass, optimum probability, feasible-shot error, X-vs-XY distribution comparison, optional lower-tail CVaR. |
| Repeatability | Fixed demo seed | Added `benchmark_qaoa.py` using matched optimizer seeds and budgets; writes CSV/JSON/PNG. |
| Stage 3 | Qiskit 6-qubit RY + MC + exact DP + synthetic readout noise | Preserved mathematical method, traceability, and validation. |
| Presentation | 10 slides | Updated 11-slide pitch with XY comparison and explicit simulation disclaimer. |

## Experiment: generated demo data only

Demo constraints: budget = 60; capacity = 12; 10 candidates; 6 selected; 22 feasible subsets among the 210 cardinality-six possibilities. The problem data is synthetic; customer IDs begin `DEMO_`.

### Single matched run

- XY: theoretical probability of a cardinality-six **and** budget/capacity-feasible bitstring: **25.35%**.
- X: the equivalent feasibility probability: **3.31%**.
- XY: ideal Hamming-weight-six probability: **100%** (a mathematical property of the ansatz).
- The final six were selected by the exact classical certificate, not by assuming QAOA obtained the optimum.

### Repeated optimizer seeds 11, 22 and 33

- XY mean feasible-state probability: **15.95%** (across those three seeds).
- X mean feasible-state probability: **2.30%**.
- The score is *not* a classical-vs-quantum speedup, not a guarantee that XY always wins, and not a result from IBM Quantum hardware.

## Quick commands

Install full dependencies and run with the real Qiskit components (on a machine with working Qiskit wheels):

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python run_pipeline.py --demo --engine qiskit --qml-backend native --qaoa-mixer xy --compare-mixers
python run_pipeline.py --demo --engine qiskit --qml-backend qiskit_ml --qaoa-mixer xy --compare-mixers
python benchmark_qaoa.py --stage2-json example/demo_outputs/stage2_selection.json --seeds 11 22 33
python verify_submission.py --out example/demo_outputs --allow-demo
```

Run fast mathematical reference without Qiskit:

```bash
python -m pip install -r requirements-reference.txt
python run_pipeline.py --demo --engine numpy --qml-backend native --compare-mixers
python benchmark_qaoa.py --stage2-json example/demo_outputs/stage2_selection.json --seeds 11 22 33
python verify_submission.py --out example/demo_outputs --allow-demo
```

For official data use the complete real-data command in `README.md`, the **official** objective, budget, capacity and column names. Set `--qml-backend qiskit_ml` explicitly to use the first requested repository; set `--engine qiskit` to generate Aer shots. `--qaoa-cvar 0.25` optionally enables lower-tail CVaR for optimization experiments (default `1.0` is regular expectation).

## Current verification state

- The self-contained NumPy reference pipeline **ran successfully**.
- The independent submission auditor **passed** on 100 synthetic customer rows and the six-customer traceability.
- Local automated tests: **11 passed, 5 skipped** (all skipped checks require `qiskit` and/or `qiskit-machine-learning`, neither installed in this build environment).
- Qiskit kernel, QSVC, Aer state preparation, transpilation and noise handling **have not been executed** here; they are implementation branches requiring installation and validation on your system.
- Before submitting, obtain and run the actual organizer files, check the detailed formula, pin the environment with `pip freeze`, upload the real GitHub project, deck and video.

## What *not* to claim

Do not claim a trained external QSVC, an executed Aer circuit, verified official customer IDs, real-world predictive performance, a quantum hardware run or a quantum advantage until each is independently demonstrated. The 22-of-210 and 25.35%-vs-3.31% figures are **synthetic simulated reference observations** only.
