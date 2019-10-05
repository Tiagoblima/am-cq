"""
Contains the class QBNN that implements the Quantum Neural Network.
"""
from itertools import combinations
import qiskit as qkit
from qiskit import Aer
import numpy as np


def generate_inputs(n_inputs):
    """Generate the inputs to the QBNN combining the positions of the quantum registers that can be changed to 1"""
    inputs = []
    regs = np.arange(n_inputs)
    for n_reg in range(n_inputs + 1):
        inputs.extend(list(combinations(regs, n_reg)))

    return inputs


def generate_labels(inputs, input_size):
    """Generate the labels to the QBNN based on the number of position change to 1"""
    labels = []
    pos = 0
    for entry in inputs:
        if len(entry) <= len(input_size) / 2:
            labels.append(pos)
            pos += 1
    return labels


class QNeuron:
    """" Builds a Quantum Neuron Circuit"""
    inputs = None
    _weights = None
    _ancillas = None
    _q_neuron = None
    output = None
    entries = []
    n_inputs = 0

    def __init__(self, n_inputs, n_weight):
        self.n_inputs = n_inputs
        self.n_weights = n_weight
        self.inputs = qkit.QuantumRegister(self.n_inputs)
        self._weights = qkit.QuantumRegister(self.n_weights)
        self._ancillas = qkit.QuantumRegister(2)
        self._output = qkit.QuantumRegister(1)
        self._q_neuron = qkit.QuantumCircuit(self._weights, self.inputs, self._ancillas, self._output)

    def get_circuit(self):
        """Returns the Q_neuron circuit"""
        return self._q_neuron

    def set_weights(self, entry=None):
        """Sets the weights manually by receiving the position that might be changed to 1"""
        print('set_weights')
        if entry is None:
            entry = []
        else:
            self._q_neuron.reset(self.inputs)

        for pos in entry:
            self._q_neuron.x(self._weights[int(pos)])

    def u_inputs(self, entry=None):
        print('u_inputs')
        """Configures the inputs setting to 1 the inputs in the position 'pos' """
        if entry is None:
            entry = []
        self._q_neuron.reset(self.inputs)
        for pos in entry:
            self._q_neuron.x(self.inputs[int(pos)])

    def u_weights(self):
        print('u_weights')
        """Apply the weights to the inputs"""
        self._q_neuron.h(self._weights)

        self._q_neuron.x(self._weights)

        for i in range(3):
            self._q_neuron.cx(self._weights[i], self.inputs[i])

        self._q_neuron.x(self._weights)
        self._q_neuron.h(self._weights)

    # self._q_neuron.barrier()

    def uf_activate(self):
        print('uf_activate')
        """Applying activation function"""

        self._q_neuron.reset(self._ancillas)

        self._q_neuron.reset(self._output)
        for inp, i in zip(list(combinations(self.inputs, self.n_inputs - 1)), range(self.n_inputs)):
            self._q_neuron.mct(inp, self._ancillas[0], self._ancillas[1])
            self._q_neuron.cx(self._ancillas[0], self._ancillas[1])
            self._q_neuron.reset(self._ancillas[0])
            if i < self.n_inputs - 1:
                self._q_neuron.x(self._ancillas[1])

        self._q_neuron.cx(self._ancillas[1], self._output[0])

        return self.get_output()

    def get_output(self):
        """" Saves the output of the Q_neuron in the Quantum register output and returns it"""
        return self._output[0]


class QBNN:
    """
    Build and run the QBNN circuit
    """
    _ancillas = None
    _q_bnn_circ = None
    output = None
    clb = None
    entries = []
    n_inputs = 0
    labels = []

    def __init__(self, entries):

        self.entries = entries
        self.n_inputs = len(entries)
        self._q_bnn_circ = qkit.QuantumCircuit()
        self.output = qkit.QuantumRegister(self.n_inputs)
        self.clb = qkit.ClassicalRegister(self.n_inputs)
        self._ancillas = qkit.QuantumRegister(2)
        self._q_bnn_circ.add_register(self._ancillas, self.output, self.clb)

    def set_label(self, labels):
        """Configures the expected label for each input"""
        print('set_label')
        if labels is None:
            labels = []

        for pos in labels:
            self._q_bnn_circ.x(self.labels[int(pos)])

    def get_circuit(self):
        """Returns the QBNN circuit"""
        return self._q_bnn_circ

    def run_circuit(self, backend_name=None):
        """Runs the QBNN circuit in a given backend or in the simulator if none backend is given"""
        print('Running...')
        self._q_bnn_circ.draw(filename='QBNN')

        if backend_name is None:
            backend_name = 'qasm_simulator'

        backend = Aer.get_backend(backend_name)
        job = qkit.execute(self._q_bnn_circ, backend)
        job_result = job.result().get_counts(self._q_bnn_circ)
        print(job_result)
        return job_result

    def save_output(self, output, train_step):
        """Saves the output in a specific qubit for each input"""

        self._q_bnn_circ.cx(output, self.output[train_step])

    def train(self):
        """Trains the QBNN circuit"""
        print('Training QBNN ', end='')
        q_neuron_1 = QNeuron(3, 3)
        q_neuron_2 = QNeuron(3, 3)

        for entry, t in zip(self.entries, range(len(self.entries))):
            print(entry)
            q_neuron_1.u_inputs(entry)
            # q_neuron_1.u_weights()
            output_1 = q_neuron_1.uf_activate()

            self._q_bnn_circ += q_neuron_1.get_circuit()

            self._q_bnn_circ.draw(filename='QBNN')

            self.save_output(output_1, t)
        # self.save_output(output_2, t)
        print('Measuring...')
        self._q_bnn_circ.measure(self.output, self.clb)
        self.run_circuit()

    def get_results(self):
        """Shows the results given by the circuit"""

        print("\nQ_BNN results (Accuracy): ")
        # self._q_bnn_circ.draw(filename='QBNN')

        clsbits = qkit.ClassicalRegister(8)
        self._q_bnn_circ.add_register(clsbits)
        self._q_bnn_circ.measure(self.output, clsbits)

        backend = Aer.get_backend('qasm_simulator')
        shots = 1024
        job_sim = qkit.execute(experiments=self._q_bnn_circ, backend=backend, shots=1024)

        circuit = self._q_bnn_circ
        result = job_sim.result()

        count = result.get_counts(circuit)

        for qub, cou in zip(count.keys(), count.values()):
            chance = (cou / shots) * 100
            print("|{}>: {:.2f}%".format(qub, chance))


inputs = generate_inputs(3)
print(inputs)
q_bnn = QBNN(inputs)

q_bnn.train()
# print(q_bnn.run_circuit())
