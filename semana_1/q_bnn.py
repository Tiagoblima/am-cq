"""
Contains the class QBNN that implements the Quantum Neural Network.
"""
from itertools import combinations
import qiskit as qkit
from qiskit import Aer
import numpy as np


class QBNN:
    """
    Build and run the QBNN
    """
    inputs = None
    _weights = None
    _ancillas = None
    _q_bnn_circ = qkit.QuantumCircuit()
    output = None
    clb = None
    entries = []

    def __init__(self):

        self.inputs = qkit.QuantumRegister(3)
        self._weights = qkit.QuantumRegister(3)
        self._ancillas = qkit.QuantumRegister(2)
        self.output = qkit.QuantumRegister(8)
        self.clb = qkit.ClassicalRegister(8)

        self._q_bnn_circ.add_register(self._weights, self.inputs, self._ancillas, self.output)

    def set_label(self):
        """Configures the expected label for each input"""
        pos = 0
        for entry in self.entries:
            if len(entry) <= len(self.inputs) / 2:
                self._q_bnn_circ.x(self.output[pos])
                pos += 1

    def generate_inputs(self):
        """Generate the inputs to the QBNN"""
        regs = np.arange(3)
        for n_reg in range(4):
            self.entries.extend(list(combinations(regs, n_reg)))

        return self.entries

    def get_circuit(self):
        """Returns the QBNN circuit"""
        return self._q_bnn_circ

    def set_weights(self, entry=None):
        """Sets the weights manually by receiving the position that might be changed to 1"""
        if entry is None:
            entry = []
        for pos in entry:
            self._q_bnn_circ.x(self._weights[int(pos)])

    def run_circuit(self, backend_name=None):
        """Runs the QBNN circuit in a given backend or in the simulator if none backend is given"""
        self._q_bnn_circ.draw(filename='QBNN')
        self._q_bnn_circ.add_register(self.clb)
        self._q_bnn_circ.measure(self.output, self.clb)

        if backend_name is None:
            backend_name = 'qasm_simulator'

        backend = Aer.get_backend(backend_name)
        job = qkit.execute(self._q_bnn_circ, backend)
        job_result = job.result().get_counts(self._q_bnn_circ)

        return job_result.get_counts(self._q_bnn_circ)

    def u_inputs(self, entry=None):
        """Configures the inputs"""
        if entry is None:
            entry = []
        for pos in entry:
            self._q_bnn_circ.x(self.inputs[int(pos)])

    def u_weights(self):
        """Apply the weights to the inputs"""
        self._q_bnn_circ.h(self._weights)

        self._q_bnn_circ.x(self._weights)

        for i in range(3):
            self._q_bnn_circ.cx(self._weights[i], self.inputs[i])

        self._q_bnn_circ.x(self._weights)

        self._q_bnn_circ.h(self._weights)

        self._q_bnn_circ.barrier()

    def uf_activate(self):
        """Applying activation function"""

        self._q_bnn_circ.ccx(self.inputs[0], self.inputs[1], self._ancillas[0])
        self._q_bnn_circ.cx(self._ancillas[0], self._ancillas[1])
        self._q_bnn_circ.reset(self._ancillas[0])
        self._q_bnn_circ.x(self._ancillas[1])
        self._q_bnn_circ.barrier()

        self._q_bnn_circ.ccx(self.inputs[1], self.inputs[2], self._ancillas[0])
        self._q_bnn_circ.cx(self._ancillas[0], self._ancillas[1])
        self._q_bnn_circ.reset(self._ancillas[0])
        self._q_bnn_circ.x(self._ancillas[1])
        self._q_bnn_circ.barrier()

        self._q_bnn_circ.ccx(self.inputs[0], self.inputs[2], self._ancillas[0])
        self._q_bnn_circ.cx(self._ancillas[0], self._ancillas[1])
        self._q_bnn_circ.cx(self._ancillas[1], self._ancillas[0])
        self._q_bnn_circ.barrier()

    def save_output(self, train_step):
        """Saves the output in a specific qubit for each input"""

        self._q_bnn_circ.cx(self._ancillas[1], self.output[train_step])
        self._q_bnn_circ.barrier()

    def train(self):
        """Trains the QBNN circuit"""
        print('Training QBNN ', end='')
        input_list = self.generate_inputs()
        self.set_label()
        t_step = 0
        for entry in input_list:
            print('#', end='')
            self.u_inputs(entry)
            self.u_weights()
            self.uf_activate()
            self.save_output(int(t_step))
            self._q_bnn_circ.reset(self.inputs)
            t_step += 1

    def get_results(self):
        """Shows the results given by the circuit"""

        self._q_bnn_circ.draw(filename='QBNN')

        clsbits = qkit.ClassicalRegister(8)
        self._q_bnn_circ.add_register(clsbits)
        self._q_bnn_circ.measure(self.output, clsbits)

        backend = Aer.get_backend('qasm_simulator')
        shots = 1024
        job_sim = qkit.execute(experiments=self._q_bnn_circ, backend=backend, shots=1024)

        circuit = self._q_bnn_circ
        result = job_sim.result()

        count = result.get_counts(circuit)
        print("\nQ_BNN results (Accuracy): ")

        for qub, cou in zip(count.keys(), count.values()):
            chance = (cou / shots) * 100
            print("|{}>: {:.2f}%".format(qub, chance))
