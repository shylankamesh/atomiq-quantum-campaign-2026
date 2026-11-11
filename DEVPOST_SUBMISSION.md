# Devpost copy-ready submission

Project title: Atomiq Quantum Campaign Optimizer

Short description: A reproducible quantum-classical term-deposit campaign using fidelity-kernel learning, fixed-cardinality XY-QAOA, exact certification and six-qubit conversion simulation.

Team members: Shylan Kamesh S, Sai Santhosh T, Derin Godson.
Team name: Shylan Kamesh S (provisional label from provided details; confirm if a distinct team name is registered).
Registered contact fields: use the separate local DEVPOST_PRIVATE_DETAILS.md.

## Technical summary
We trained and validated on 300 official labeled customers and predicted 100 hidden-label customers. Three-fold training OOF validation selected a 4-qubit FidelityQuantumKernel/QSVC, fitted on a disclosed 64-row stratified training subset after final training-only preprocessing. We apply the official alphanumeric tie rule, optimize expected net value over the top ten, and certify exactly six selections by exhaustive enumeration. Ten-qubit XY-QAOA produces actual Aer measurements, while the final policy explicitly uses the exact classical certificate. A six-qubit RY circuit estimates P(at least three subscriptions), benchmarked against exact enumeration and Monte Carlo, including readout-noise mitigation.

## Required computational fields
Six selected customer IDs: C0005, C0007, C0009, C0020, C0022, C0036
Stage 3 quantum probability: 0.022796630859375
If the form accepts only three decimal places: 0.023
SDK: Qiskit 2.6.2; Qiskit Machine Learning 0.9.1; Qiskit Aer 0.17.2.
Backends: Qiskit StatevectorSampler for kernel, AerSimulator for QAOA and campaign shots.
Qubits: Stage 1 4; Stage 2 10; Stage 3 6.
Depth: feature map 5 logical / 11 u-cx; XY-QAOA 44 logical / 6144 u-cx; Stage 3 2.
Shots: kernel 1024 per fidelity; main QAOA 8192; Stage 3 32768 ideal and 32768 noisy.
Budget: 9/25. Capacity: 19/22. Expected net value: 6.13751192496118.
Classical optimum: 6.13751192496118. Measured QAOA gap: 0. Final hybrid gap: 0.

GitHub: https://github.com/shylankamesh/atomiq-quantum-campaign-2026
Presentation: https://github.com/shylankamesh/atomiq-quantum-campaign-2026/blob/main/slides/Atomiq_Official_Pitch.pdf

## Integrity declarations supported by evidence
- Training-only preprocessing and model selection; no hidden test labels.
- Frozen continuous probabilities passed through all stages.
- Official top-ten rule, exactly six feasible customers, exact certificate.
- Mandatory results at repository root, seeds and exact package versions recorded.
- Quantum execution is simulator execution. No hardware or speedup claim.
- 44 tests passed, zero skipped; independent result verifier passed.
- OOF performance is modest, and 86 test customers tie at the top probability; this is disclosed.

## Team checks before final submission
Confirm prior Google Form registration, registered team name and member accounts. Open the public repository and PDF link, paste fields into Devpost, review the integrity declarations and submit before 10:00 PM IST on 11 November 2026. Publication of the repository alone does not submit the Devpost entry.
