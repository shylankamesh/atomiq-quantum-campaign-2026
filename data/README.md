# Organizer data

Run `python fetch_data.py` to download and SHA-256 verify the public participant workbook.

Source: https://drive.google.com/drive/folders/1vjIM7mD7XjUjvfotAXF4uVfE2cYTS7Sp

Workbook: bank_campaign_participant_release.xlsx. SHA-256: d2f8e63705af2883c7618c463a6ac2f351119d59a8869d2677f8825148e9a709

Named sheets: README, Stage1_Train, Stage1_Test, Stage2_Data, Stage2_Constraints. Training: 300, test: 100. Objective p*expected_deposit_value-contact_cost, N=10, K=6, budget=25, capacity=22. Tie order: ascending alphanumeric ID.

The workbook is excluded from public Git because its redistribution license is unspecified. The MIT source license does not relicense organizer data. Never seek or add hidden test labels.
