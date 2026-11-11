# Clean public checkout verification

Public main was cloned with core.autocrlf=false to test preservation of the exact JSON/CSV bytes. fetch_data.py downloaded the official workbook and matched SHA-256 d2f8e63705af2883c7618c463a6ac2f351119d59a8869d2677f8825148e9a709.

verify_submission.py: PASS, official-data mode.

Final clean-checkout pytest command (outside the desktop execution sandbox, with a fresh workspace-local temporary directory):

```bash
python -m pytest -q --basetemp=../clean_test_temp
```

Result: 44 passed, 0 failed, 0 skipped in 18.27 seconds.

Earlier clean-checkout attempts are not counted as passes: the sandboxed run had 42 passed and 2 failed with an Aer Windows native DLL initialization error (controller_wrappers, 0xc0000008). The first unsandboxed run had 30 passed and 14 setup errors because the existing shared pytest temporary directory was inaccessible. Running outside the sandbox with a new local temporary directory resolved both in the final clean-checkout verification. The exact root cause of the intermittent native-loader behavior was not established; do not describe it as a code fix or universally resolved Windows issue.

If the same Windows native-loader error appears, close the failed interpreter and run the pinned environment in a normal terminal. First check `python -c "import qiskit_aer; print(qiskit_aer.__version__)"`, then run pytest with a fresh writable temporary directory. Do not skip Aer tests or substitute NumPy outputs. No security settings or system libraries were changed.

Public PDF was downloaded without authentication and its SHA-256 matched the reviewed local deck: 486b2cb5e3f6ec78501ddda667d513a5005eb6f607091c4f81e6547c74d1d779.
