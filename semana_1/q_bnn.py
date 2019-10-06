"""
Contains the class QBNN that implements the Quantum Neural Network.
"""
from itertools import combinations
import qiskit as qkit
from qiskit import Aer, QuantumRegister, ClassicalRegister, QuantumCircuit
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


class Layer:
    """" Builds a Quantum Neuron Circuit"""
    inputs = None
    _weights = None
    _ancillas = None
    _q_neuron = None
    _output = None
    entries = []
    n_inputs = 0

    def __init__(self, circuit, q_weight, q_inputs, n_output):

        self.n_inputs = len(q_inputs)
        self.inputs = q_inputs
        self._weights = q_weight
        self._output = qkit.QuantumRegister(n_output)
        circuit.add_register(self._output)

        self._q_neuron = circuit

    def set_ancillas(self, q_ancillas):
        self._ancillas = q_ancillas

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

        self._q_neuron.reset(self._weights)
        self._q_neuron.h(self._weights)

        self._q_neuron.x(self._weights)

        for i in range(self.n_inputs):
            self._q_neuron.cx(self._weights[i], self.inputs[i])

    def set_input_rg(self, regs):
        self.inputs = regs
        self._q_neuron.add_register(regs)

    def u_activate(self, output_reg):
        qr = QuantumRegister(3)
        cr = ClassicalRegister(2)

        circuit_1 = QuantumCircuit(qr, cr)
        self._q_neuron.add_register(qr)
        self._q_neuron.ccx(self.inputs[0], self.inputs[1], self._output[0])
        get_results(self._q_neuron, 1, qr[2])

    def uf_activate(self, output_reg):
        print('uf_activate')
        """Applying activation function"""
        i = 0
        self._q_neuron.reset(self._ancillas)
        inps = list(combinations(self.inputs, 1))
        for inp in inps:
            self._q_neuron.x(inp[0])
            self._q_neuron.mct(self.inputs, self._output[output_reg], self._ancillas)
            self._q_neuron.x(inp[0])

        self._q_neuron.mct(self.inputs, self._output[output_reg], self._ancillas)

    # self._q_neuron.cx(self._ancillas[1], self._output[output_reg])

    def get_output(self):
        """" Saves the output of the Q_neuron in the Quantum register output and returns it"""
        return self._output


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
    inputs = None

    def __init__(self, entries, input_size):

        self.entries = entries
        self.n_inputs = len(entries)

        self._ancillas = qkit.QuantumRegister(2)
        self.output = qkit.QuantumRegister(self.n_inputs)
      #  self.clb = qkit.ClassicalRegister(self.n_inputs)
        self.inputs = qkit.QuantumRegister(input_size)
        self.weights = qkit.QuantumRegister(input_size)
        self._q_bnn_circ = qkit.QuantumCircuit(self.weights, self.inputs, self._ancillas, self.output)

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
        inputs = generate_inputs(3)
        print(inputs)
        q_input = qkit.QuantumRegister(3)

        layer_1 = Layer(self._q_bnn_circ, self.weights, self.inputs, 2)
        layer_1.set_ancillas(self._ancillas)
        t_step = 0
        for inp in inputs:
            layer_1.u_inputs(inp)
            # layer_1.u_weights()
            # First Layer

            # First neuron

            layer_1.uf_activate(0)

            # Second Neuron

            layer_1.uf_activate(1)

            self._q_bnn_circ.barrier()

            layer_2 = Layer(self._q_bnn_circ, self.weights, layer_1.get_output(), 1)
            layer_2.set_ancillas(self._ancillas)

            # layer_2.u_weights()
            layer_2.uf_activate(0)
            self.save_output(layer_2.get_output(), t_step)
            t_step += 1

    def get_output(self):
        return self.output


def get_results(circ, n_clb, output):
    """Shows the results given by the circuit"""

    print("\nQ_BNN results (Accuracy): ")
    circ.draw(filename='QBNN')

    clsbits = qkit.ClassicalRegister(len(output))
    circ.add_register(clsbits)
    circ.measure(output, clsbits)

    backend = Aer.get_backend('qasm_simulator')
    shots = 1024
    job_sim = qkit.execute(experiments=circ, backend=backend, shots=1024)

    result = job_sim.result()

    count = result.get_counts(circ)

    for qub, cou in zip(count.keys(), count.values()):
        chance = (cou / shots) * 100
        print("|{}>: {:.2f}%".format(qub, chance))
