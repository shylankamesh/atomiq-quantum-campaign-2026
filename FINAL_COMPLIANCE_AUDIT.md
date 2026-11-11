# Final compliance audit

Audit date: 11 November 2026. Authorities: four-page official problem statement, organizer workbook README/constraints, and supplied Devpost rules. Technical implementation verified; final Devpost submission is a separate team action.

| Requirement | Evidence | Verification | Observed result | Status / remaining action |
|---|---|---|---|---|
| 300 training / 100 hidden-label test rows | organizer workbook, run_manifest.json | fetch_data.py; schema test | 300/100; 35 training positives; SHA-256 recorded | PASS |
| No test leakage | stage1.py; validation folds | test_model_selection_independent_of_test_features | Changed test values leave metrics/model choice unchanged | PASS |
| Quantum and classical probabilistic classifiers | stage1_validation.json, fitted model | official pipeline execution | Genuine FidelityQuantumKernel/QSVC, logistic regression and histogram gradient boosting | PASS |
| Training-only validation/calibration | stage1.py | three-fold OOF evidence | All outer transforms fit within training folds; QSVC inner calibration shares outer transforms, disclosed | PASS with stated limitation |
| Final refit | stage1_validation.json | training_fit_samples and quantum subset IDs | Final preprocessing/classical fit 300; disclosed quantum subset 64 | PASS, subset limitation disclosed |
| Continuous probabilities and correct ranking | stage1_predictions.csv | verify_submission.py | 100 unique IDs, ranks 1..100, 10 flags, descending alphanumeric ties | PASS |
| Freeze Stage 1 before downstream use | runner; CSV hashes | verifier and tamper tests | Same probabilities in CSV, Stage2 inputs, Stage3 mapping | PASS |
| Official Stage 2 objective | stage2_selection.json | independently reconstruct p*V-c | Expected net value 6.13751192496118 | PASS |
| Exactly six from top ten | root result files | independent verifier | C0005,C0007,C0009,C0020,C0022,C0036 | PASS |
| Budget and capacity | Stage2_Data/Constraints | compare workbook values and sums | Cost 9<=25; capacity 19<=22 | PASS |
| Exact benchmark | stage2_all_210_subsets.csv | independent combination enumeration | All 210 feasible; optimum independently matches | PASS |
| Hamiltonian, penalties, Ising distinction | stage2.py; qubo.py; qubo_coefficients.json | energy-sign and algebra tests | Executed pseudo-Boolean oracle; separate 20-variable quadratic alternative validated | PASS |
| Quantum optimizer and actual counts | stage2_selection.json | Aer execution; count reconstruction | 8192 XY shots; all weight six; measured gap zero | PASS |
| Quantum versus classical boundary | final_selection_policy | inspect results/README | Exact certificate provides final choice; separate measured/final gaps | PASS |
| Repeated optimization trials | qaoa_repeated_trials.json/csv | 12 genuine Aer trials | seeds 11/22/33, p=1/2, X/XY; retained counts | PASS |
| Exactly six RY qubits | stage3_simulation.json; diagrams | Aer and basis-state tests | 6 qubits, depth 2, rightmost=q0 | PASS |
| At least three subscriptions | stage3 counts | independent 64-outcome enumeration/counting | 747/32768 = 0.022796630859375 | PASS |
| Exact and Monte Carlo references | stage3_simulation.json | DP vs independent enumeration | Exact 0.02404929745931035; MC 0.02365 | PASS |
| Ideal/noisy/mitigated estimates | stage3_simulation.json | ideal/noisy Aer runs | 0.02279663 / 0.03445435 / 0.02241992 | PASS; synthetic readout-only model |
| Error propagation and uncertainty | sensitivity_analysis.json | 200 rerank/reselect perturbations; Wilson interval | Selection instability and conditional shot uncertainty disclosed | PASS; sensitivity is not a confidence bound |
| Logical/physical counts and transpilation | circuit metrics JSON files | u/cx synthesis | QAOA logical depth 44, basis depth 6144; simulation identity layout | PASS |
| Exact dependencies and seeds | requirements*, run_manifest.json | pip check; recorded versions | No broken requirements; actual seeds recorded | PASS |
| Working tests | docs/TEST_RESULTS.txt | python -m pytest -q | 44 passed, 0 failed, 0 skipped | PASS |
| Independent tamper detection | test_verifier_tampering.py | changed objectives/probabilities/count metrics/flags | Invalid bundles rejected without relying on hashes | PASS |
| Root-level results | three mandatory root files | verify_submission.py | PASS official-data mode | PASS |
| Technical report/business interpretation | README.md, FINAL_EXPERIMENT_REPORT.md | factual cross-check against JSON | Modest classifier quality, ties and 2.4% modeled success disclosed | PASS |
| License/reference credits | LICENSE, README | source inspection | MIT own source; no copied GPL code; data not relicensed | PASS |
| Public source repository | GitHub URL in README | remote commit/file verification | Published to public main branch; remote verification in PUBLISH_STATUS.md | PASS |
| Presentation | slides/Atomiq_Official_Pitch.pptx/pdf | 12 rendered slides visually inspected; package finalizer | Editable PPTX and view-only PDF prepared | PASS; public PDF hosted in repository |
| Team fields | DEVPOST_SUBMISSION.md; separate local private details | user-provided names/emails/handles | Three members provided; exact registered team label needs confirmation | ACTION: team confirm label |
| Prior registration/eligibility | team records | user check required | Cannot establish prior Google Form registration from project | BLOCKED on team confirmation |
| Submit via Devpost by 10 PM IST | Devpost account | final submission receipt | No submission receipt obtained | BLOCKED on final team submission |

## Scientific limitations

The selected QSVC has pooled OOF average precision 0.162837 and ROC-AUC 0.549057. The final four selected categorical indicators map 86 test rows to the top tied probability. The official tie rule is followed exactly. No claim of strong real-world discrimination, quantum advantage, hardware execution or overall competition readiness is justified by this alone. The main oracle is general diagonal, and its basis-gate depth is expensive. Stage 3 assumes independent outcomes and uses synthetic readout noise with known correction matrix.

No mandatory scientific execution remains unverified. Account-level eligibility and the final Devpost submission remain outside the local technical checks.
