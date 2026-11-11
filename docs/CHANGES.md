# Changes from the supplied archive

- run_pipeline.py: official workbook/separate metadata support, expected-profit default, full-precision frozen propagation, model save, hash-checked resume, provenance and runtime.
- quantum_campaign/data.py: yes/no labels, named-sheet loading, validated ID joins, train/test disjointness, leakage exclusions.
- quantum_campaign/stage1.py: three-fold training OOF selection, final refit, seeded quantum subset, calibration diagnostics and model persistence.
- quantum_campaign/qml_qiskit_ml.py: explicit seeded Qiskit StatevectorSampler with FidelityQuantumKernel/QSVC.
- quantum_campaign/stage2.py: certified oracle penalty, input validation, complete enumeration, value fields and basis metrics.
- quantum_campaign/qubo.py: independent true quadratic slack/Ising encoding and coefficients.
- quantum_campaign/circuit_metrics.py: separate Aer-native and u/cx metrics.
- quantum_campaign/stage3.py: validation, Wilson interval, successful-shot totals, seeds, bit-order and gate metrics.
- verify_submission.py: independent 210/64 enumeration, workbook comparisons, probability/objective/count/gap checks, hash integrity.
- complete_evidence.py: repeated genuine Aer trials, sensitivity and actual circuit/calibration figures.
- fetch_data.py: official public workbook retrieval with SHA-256 verification.
- tests/test_submission_correctness.py and tests/test_verifier_tampering.py: meaningful independent and tampering tests.
- requirements.txt / requirements-lock.txt: actual installed versions.
- README.md, START_HERE.md, SUBMISSION_CHECKLIST.md and data/README.md: current verified official workflow.
- docs/INITIAL_AUDIT.md, FINAL_COMPLIANCE_AUDIT.md, FINAL_EXPERIMENT_REPORT.md: source audit and actual observed evidence.
- DEVPOST_SUBMISSION.md: copy-ready public computational fields; private member contacts kept outside repository.
- slides/Atomiq_Official_Pitch.pptx/pdf: updated supplied deck with actual evidence and 12 inspected slides.
- Root result/evidence JSON/CSV and figures: genuine official-data outputs.
- Historical UPGRADE_NOTES.md and DEMO_VIDEO_SCRIPT.md retained but clearly marked superseded synthetic material.

Original archive preserved in the local work directory before edits. No GPL implementation code copied.
