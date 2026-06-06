from __future__ import annotations

from qiskit import QuantumCircuit

from compiler_backend.heron import (
    CompileReport,
    add_measurements,
    select_calibration_aware_layout,
)


class _FakeCouplingMap:
    def __init__(self, edges):
        self._edges = edges

    def get_edges(self):
        return list(self._edges)


class _FakeConfig:
    n_qubits = 6


class _FakeProperties:
    def readout_error(self, qubit):
        return {0: 0.001, 1: 0.001, 2: 0.002, 3: 0.02, 4: 0.03, 5: 0.04}.get(qubit, 0.05)

    def gate_error(self, op_name, qargs):
        edge = tuple(qargs)
        low = {(0, 1), (1, 0), (1, 2), (2, 1)}
        return 0.001 if edge in low else 0.02


class _FakeBackend:
    name = "fake_heron"
    coupling_map = _FakeCouplingMap([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    processor_type = {"family": "Heron", "revision": "r3"}

    def configuration(self):
        return _FakeConfig()

    def properties(self):
        return _FakeProperties()


def test_add_measurements_target_only():
    qc = QuantumCircuit(3)
    qc.x(1)

    measured = add_measurements(qc, qubits=[1])

    assert qc.num_clbits == 0
    assert measured.num_clbits == 1
    assert measured.count_ops().get("measure", 0) == 1


def test_select_calibration_aware_layout_prefers_low_error_patch():
    layout = select_calibration_aware_layout(_FakeBackend(), n_qubits=3)

    assert layout == [0, 1, 2]


def test_compile_report_as_dict():
    report = CompileReport(
        backend_name="ibm_boston",
        processor_type="Heron r3",
        n_qubits=23,
        depth=100,
        two_qubit_depth_proxy=20,
        gate_counts={"ecr": 20},
        selected_physical_qubits=[0, 1, 2],
        estimated_success_probability=0.9,
        notes="ok",
    )

    d = report.as_dict()
    assert d["backend_name"] == "ibm_boston"
    assert d["gate_counts"] == {"ecr": 20}


def test_multivariate_gci_builds_with_internal_normal_fallback():
    from Code.multivariateGCI import MultivariateGCI_Linear

    circuit = MultivariateGCI_Linear(
        n_normal=2,
        normal_max_value=2,
        p_zeros=[0.01, 0.02],
        rhos=[0.09, 0.09],
        F_list=[[0.1, 0.2], [0.0, 0.1]],
    )

    assert circuit.num_qubits == 6
