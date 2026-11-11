> Historical synthetic-demo material. Superseded for official results by README.md and FINAL_EXPERIMENT_REPORT.md. Video is optional under the supplied rules.

# 1–2 minute Devpost demonstration script (~105 seconds)

> Replace **every placeholder with your actual verified Qiskit run**. Do not present the synthetic demo as an official outcome.

**0–12s: Hook (Slide 1)**
"How can a business find high-conversion customers when budget and capacity are limited? We built a transparent three-stage quantum–classical marketing pipeline."

**12–30s: Stage 1 (Slide 4 / terminal output)**
"Four-qubit entangled feature maps turn customer attributes into a quantum fidelity kernel. We compare Qiskit Machine Learning FidelityQuantumKernel/QSVC or an equivalent statevector reference against logistic regression and gradient boosting, select a blend weight on held-out validation, and rank all 100 test customers. Here's our validation PR-AUC and Brier score: [REAL METRICS]."

**30–56s: Stage 2 (Slide 5 / Stage 2 JSON; optionally Slide 10)**
"Next, a ten-qubit QAOA phase oracle uses a weight-six initial state and an XY mixer that preserves exactly six selected people. Budget and capacity are checked separately. We benchmark it against an unconstrained X-mixer and a classical exact search over 210 six-customer choices. Our measured QAOA optimality gap is [REAL GAP]. The final choice [REAL SIX IDS] is classically verified, and we make no quantum-advantage claim."

**56–81s: Stage 3 (Slide 6 / Stage 3 JSON)**
"Six qubits encode six predicted probabilities via RY rotations. Success is at least three measured ones. We compare Aer shot counts against the exact Poisson-binomial calculation and Monte Carlo. We then add independent readout errors and apply a calibrated inversion. Our ideal estimate is [REAL P], exact is [REAL EXACT], and shot error is [REAL ERROR]."

**81–105s: Close (Slides 8–11)**
"This is fully reproducible: fixed seeds, test-leakage safeguards, public code and root-level outputs. Quantum contributions are separated clearly from classical certification. The circuit has a costly generic phase oracle and state preparation; next steps include hardware-efficient decomposition and device-calibrated noise."

## Capture checklist

- 1080p screen recording; large terminal text, no private credentials, under 2 minutes.
- Show `python run_pipeline.py --... --engine qiskit` finishing successfully with actual data.
- Show the actual Qiskit feature map and conversion circuit, plus results/plots.
- Show `pytest` passing, with Qiskit-specific tests **not skipped**.
- Upload to YouTube (unlisted) or accessible Drive/Loom and test the link while logged out.
