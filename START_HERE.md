# Start here

The verified official results are at the repository root. Read README.md and FINAL_COMPLIANCE_AUDIT.md.

```bash
python -m pip install -r requirements.txt
python fetch_data.py
python verify_submission.py
python -m pytest -q
```

To reproduce the complete official run:

```bash
python run_pipeline.py --workbook data/bank_campaign_participant_release.xlsx --engine qiskit --qml-backend qiskit_ml --compare-mixers --qaoa-shots 8192 --simulation-shots 32768
```

Use DEVPOST_SUBMISSION.md for fields. A public repository is not a completed Devpost submission.
