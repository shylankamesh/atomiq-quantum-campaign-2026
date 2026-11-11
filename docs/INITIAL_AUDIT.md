# Initial audit, 11 November 2026

Authority: all four rendered pages of the supplied problem PDF, organizer workbook README and Stage2_Constraints sheets, and supplied/online Devpost rules. Original archive preserved separately before edits.

| Requirement | Existing source/function | Observed defect or status | Correction / verification |
|---|---|---|---|
| Official 300/100 data | data.validate_inputs, run_pipeline.table | Default Excel reader reads README; yes/no target rejected; separate Stage2_Data unsupported | Read named workbook sheets, normalize training labels only, validated ID join and exact row counts |
| Training-only preprocessing | stage1.train_and_predict | Holdout preprocessing uses training fold, but test predictions computed during selection and final fit only 75% | Cross-validated training predictions, freeze selection, refit on all 300 before test inference |
| Quantum/classical classifiers | qml_qiskit_ml.fit_predict_qsvc | Actual FidelityQuantumKernel/QSVC pathway exists; not yet executed, sampler not explicitly seeded | Install isolated dependencies, seed primitive, run genuine classifier and tests |
| Continuous ranking / ties | stage1.train_and_predict | Alphanumeric string sort matches organizer README; preserve | Test ties, output schema, identity |
| Expected net value | run_pipeline.parser / stage2.solve | Defaults to conversions, contrary to official p*value-cost | Default expected_profit and read official value column |
| Exactly six, budget 25, capacity 22 | stage2.exact_solve | Enumerates combinations, valid check; runner has no official workbook handling | Read constraints; save all 210 rows and independent certificate |
| Quantum formulation | stage2.energy_table | Correctly calls piecewise oracle pseudo-Boolean, but arbitrary penalties not proved sufficient | Certified penalty dominance and executable quadratic slack alternative |
| Fixed-weight XY | advanced_qaoa.xy_qiskit_circuit | Independent implementation matches claimed XX/YY mixer; prior Qiskit tests skipped | Run equivalence and measurement tests with installed Qiskit |
| Quantum/classical separation | stage2.solve | Final exact solution transparently disclosed | Preserve distinct measured and final gaps; verify from counts |
| Six-qubit campaign | stage3.simulate | Correct RY encoding and Poisson-binomial DP by inspection; not yet executed | Genuine Aer ideal/noisy runs, counts/angles/mapping checks |
| Frozen probabilities | run_pipeline.main | Comment says persisted but uses in-memory values after rounded CSV write | Persist full precision, reload canonical probabilities for stages 2/3 |
| Noise and uncertainty | stage3.simulate | Readout noise and inverse correction present; no interval, few circuit metrics | Wilson shot interval, circuit depth/gate metrics, sensitivity experiment |
| Independent verifier | verify_submission.check | Reuses solver functions; does not reconstruct p*value-cost or validate all counts/gaps | Independent enumeration and probability enumeration; tamper tests |
| Reproducibility | requirements.txt, manifest | Version ranges and unexecuted quantum branches | Freeze installed versions; record hashes, all seeds, runtime/backend |
| Root results and report | runner / README | Only synthetic prior evidence | Real organizer run, verified root outputs and reports |
| Presentation | slides and supplied PPTX | Demo content, no official results | Update existing deck with measured evidence and inspect rendered pages |
| Public links and team | submission checklist | Not present in supplied project | Prepare copy-ready Devpost fields; public URLs/team require available account/user details |

This is an initial source audit, not a declaration of passed execution. Final observed results belong in FINAL_COMPLIANCE_AUDIT.md.
