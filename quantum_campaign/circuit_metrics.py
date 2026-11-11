"""Explicit u/cx synthesis metrics, separately from Aer-native simulator depth."""
def basis_metrics(circuit, seed=42):
    from qiskit import transpile
    compiled = transpile(circuit, basis_gates=['u', 'cx'], optimization_level=1, seed_transpiler=seed)
    operations = list(compiled.data)
    return dict(basis_gates=['u','cx'], basis_transpiled_depth=compiled.depth(),
                basis_gate_counts=dict(compiled.count_ops()),
                one_qubit_gate_count=sum(len(x.qubits)==1 and x.operation.name not in ('measure','reset','barrier') for x in operations),
                two_qubit_gate_count=sum(len(x.qubits)==2 for x in operations),
                swap_count=int(compiled.count_ops().get('swap',0)),
                logical_to_simulator_qubits=list(range(circuit.num_qubits)),
                physical_layout_note='Unconstrained simulator topology; no hardware device placement')
