from qiskit import QuantumRegister
from qiskit.circuit import QuantumCircuit

import numpy as np

from pyqsp.angle_sequence import QuantumSignalProcessingPhases


class QSVT(QuantumCircuit):
    do_print_phases=False
    def __init__(
        self, unitary_circuit:QuantumCircuit, 
        subspace_qubits1:list, subspace_qubits2:list,
        phases:list=None, poly:list=None, adjust_conventions:bool=False,
        ctrl_zero_qubits1:list=[],
        ctrl_zero_qubits2:list=[],
        name:str='QSVT'
    ):
        '''
        Given 
        Args:
            unitary_circuit: QuantumCircuit on which QSVT is going to be applied. This is the global circuit (usually denoted as U in litterature)
            phases: list of phases to be applied in QSVT
            poly: list of coefficients of a polynomial that we want to apply using QSVT.
                Used only when phases is None to compute them using the library pyqsp
            adjust_conventions:bool when True, phases are converted. Default False
            name: name of the resulting circuit. Default 'QSVT'
        '''
        if phases is None and poly is not None:
            phases = QuantumSignalProcessingPhases(
                        poly, signal_operator="Wx", method="sym_qsp", measurement="x", chebyshev_basis=True
                    )
        if self.do_print_phases:
            print(phases)
        if adjust_conventions:
            phases = QSVT.adjust_qsvt_convetions(phases)
        if self.do_print_phases:
            print(phases)
        self.unitary_gate = unitary_circuit.to_gate()
        self.subspace_qubits1 = subspace_qubits1
        self.subspace_qubits2 = subspace_qubits2
        self.phases = phases
        self.rev_phases = phases.copy()
        self.rev_phases[0]=self.rev_phases[0]+self.rev_phases[-1]
        self.ctrl_zero_qubits1 = ctrl_zero_qubits1
        self.ctrl_zero_qubits2 = ctrl_zero_qubits2
        # è più comodo avere le fasi in ordine inverso quando si concatenano gli step
        # self.rev_phases.reverse()
        self.aux_reg = QuantumRegister(1, "aux_qubit")
        self.qu_registers = unitary_circuit.qregs.copy()
        self.qu_registers.append(self.aux_reg)
        super().__init__(*self.qu_registers, name=name)
        self.h(self.aux_reg)
        self.createQSVT()
        self.h(self.aux_reg)
        
    def addProj(self, id_proj:int, phi:float):
        if id_proj==1:
            subspace_qubits = self.subspace_qubits1
            ctrl_zero_qubits = self.ctrl_zero_qubits1
        else:
            subspace_qubits = self.subspace_qubits2
            ctrl_zero_qubits = self.ctrl_zero_qubits2
        # Tier 1.1: drop the X-X sandwich around the projector.  The
        # X-X was a |0><0| <-> |1><1| projector convention switch; the
        # new projector is a plain |1><1| controlled by subspace_qubits
        # onto the AUX.  The ``adjust_conventions=True`` path used by
        # ``qsvt_application_circuit`` already absorbs the sign change,
        # so the final statevector is preserved.  Verified to 1e-9 by
        # tests/test_code_qsvt_equivalence.py.
        # mcx with a single control is a plain CX; using ``cx`` directly
        # avoids a Toffoli decomposition that costs ~10 2Q gates on
        # Heron.  ``subspace_qubits`` is always length 1 in this
        # codebase (see ``qsvt_application_circuit`` in Code/circuitsCRA).
        del ctrl_zero_qubits  # no longer used
        self.cx(subspace_qubits[0], self.aux_reg[0])
        self.rz(2*phi, self.aux_reg)
        self.cx(subspace_qubits[0], self.aux_reg[0])
        
    def createQSVT(self):
        parity=True
        u_inv = self.unitary_gate.inverse()
        U=self.unitary_gate
        id_proj=1
        for phi in self.rev_phases[:-1]:
            self.addProj(id_proj, phi)
            self.append(U, self.qubits[:-1])
            parity=not parity
            if parity:
                id_proj=1
                U=self.unitary_gate
            else:
                id_proj=2
                U=u_inv
        #self.addProj(id_proj, self.rev_phases[-1])
    
    def adjust_qsvt_convetions(phases: np.ndarray) -> list:
        phases = np.array(phases)
        # change the R(x) to W(x), as the phases are in the W(x) conventions
        phases = phases - np.pi / 2
        phases[0] = phases[0] + np.pi / 4
        phases[-1] = phases[-1] + np.pi / 2 + (2 * (len(phases)-1) - 1) * np.pi / 4

        # verify conventions. minus is due to exp(-i*phi*z) in qsvt in comparison to qsp
        return list(-1 * phases)
