"""
Contains the class QBNN that implements the Quantum Neural Network.
"""
from itertools import combinations
from qiskit import Aer, QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute, exceptions
import numpy as np


def generate_inputs(inputs_size):
    """Generate the inputs to the QBNN combining the positions of the quantum registers that can be changed to 1"""
    inputs = []
    regs = np.arange(inputs_size)
    for n_reg in range(inputs_size + 1):
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
    n_outputs = 0

    def __init__(self, circuit, q_weight, q_inputs):

        self.n_inputs = len(q_inputs)
        self.inputs = q_inputs
        self._weights = q_weight
        
        self._q_neuron = circuit

    def twoqubits_activation(self, output_reg):
        
        self._q_neuron.ccx(self.inputs[0], self.inputs[1], self._output[output_reg])
        self._q_neuron.cx(self.inputs[0], self._output[output_reg])
        self._q_neuron.cx(self.inputs[1], self._output[output_reg])

    def threequbits_activation(self, output_reg):
        
        self._q_neuron.ccx(self.inputs[0], self.inputs[1], self._output[output_reg])
        self._q_neuron.ccx(self.inputs[1], self.inputs[2], self._output[output_reg])
        self._q_neuron.ccx(self.inputs[0], self.inputs[2], self._output[output_reg])
    
    def large_activation(self, output_reg):

        self._q_neuron.reset(self._ancillas)
        inps = list(combinations(self.inputs, 1))
        for inp in inps:
            self._q_neuron.x(inp[0])
            self._q_neuron.mct(self.inputs, self._output[output_reg], self._ancillas)
            self._q_neuron.x(inp[0])

        self._q_neuron.mct(self.inputs, self._output[output_reg], self._ancillas)

    def set_ouput(self, q_ouput):
        self.n_outputs = len(q_ouput) # It is also the quantity of required neurons 
        self._output = q_ouput
        try:
            
            self._q_neuron.add_register(self._output)
        except exceptions.QiskitError:
            pass

    def reset_output(self):
     
      self._q_neuron.reset(self._output)

    def set_ancillas(self, q_ancillas):
        self._ancillas = q_ancillas

    def set_output_reg(self, output_reg):
        self._output = output_reg

    def get_circuit(self):
        """Returns the Q_neuron circuit"""
        return self._q_neuron

    def set_weights(self, entry=None):
        """Sets the weights manually by receiving the position that might be changed to 1"""
        print('setting weights')
        if entry is None:
            entry = []
        else:
            self._q_neuron.reset(self.inputs)

        for pos in entry:
            self._q_neuron.x(self._weights[int(pos)])

    def u_inputs(self, entry=None):
     
        """Configures the inputs setting to 1 the inputs in the position 'pos' """
        if entry is None:
            entry = []
       
        for pos in entry:
            self._q_neuron.x(self.inputs[int(pos)])

    def u_weights(self):

        """Apply the weights to the inputs"""

        self._q_neuron.x(self._weights)

        for i in range(self.n_inputs):
            self._q_neuron.cx(self.inputs[i], self._weights[i])
        self._q_neuron.x(self._weights)
        
    def set_input_rg(self, regs):
        self.inputs = regs
        self._q_neuron.add_register(regs)

    def apply_neurons(self):
        
        for neuron in range(self.n_outputs):
            self.uf_activate(neuron)

    def uf_activate(self, output_reg):

        """Applying activation function"""
        print(len(self.inputs))
        if len(self.inputs) is 2:
            self.twoqubits_activation(output_reg)
        elif len(self.inputs) is 3:
            self.threequbits_activation(output_reg)
        else:
            self.large_activation(output_reg)
        
    def get_output(self):
        """" Saves the output of the Q_neuron in the Quantum register output and returns it"""
        return self._output


class QBNN:
    """
    Build and run the QBNN circuit
    """
    _ancillas = None
    _q_bnn_circ = None
    clb = None
    entries = []
    n_inputs = 0
    labels_set = []
    inputs = None
    _output = None 
    l_outs = []


    def __init__(self, entries, input_size, n_layers):

        self.entries = entries
        self.n_inputs = len(entries)
        self.n_layers = n_layers
        self._ancillas = QuantumRegister(2)
        self.outputs = QuantumRegister(self.n_inputs, name='out')
        self.inputs = QuantumRegister(input_size, name='i')
        self.weights = QuantumRegister(input_size, name='w')
        self._q_bnn_circ = QuantumCircuit(self.weights, self.inputs, self.outputs)
        
        self.l_outs = []

        for l_layer in range(n_layers, 0, -1):
            self.l_outs.append(QuantumRegister(l_layer, name='l'+str(n_layers-l_layer)+'_out'))

    def set_label(self, labels_set=None):
        """Configures the expected label for each input"""

        if labels_set is None:
            labels_set = []

        for pos in labels_set:
            self._q_bnn_circ.x(self.outputs[int(pos)])

    def get_circuit(self):
        """Returns the QBNN circuit"""
        self._q_bnn_circ.draw(filename='QBNN')
        return self._q_bnn_circ

    def run_circuit(self, backend_name=None):
        """Runs the QBNN circuit in a given backend or in the simulator if none backend is given"""
        print('Running...')

        if backend_name is None:
            backend_name = 'qasm_simulator'

        backend = Aer.get_backend(backend_name)
        job = execute(self._q_bnn_circ, backend, shots=1)
        job_result = job.result().get_counts(self._q_bnn_circ)
        print(job_result)
        return job_result

    def save_output(self, output_reg):
        """Saves the output in a specific qubit for each input"""
        self._q_bnn_circ.cx(self._output, self.outputs[output_reg])

    def _run_qbnn(self, q_inputs, auto_weights):
       
        l_counter = 2

        q_input = self.inputs
        l_counter = self.n_layers
        for l_out in self.l_outs:
            
            layer = Layer(self._q_bnn_circ, self.weights, q_input)
            layer.set_ouput(l_out)

            if l_counter is self.n_layers:
                layer.u_inputs(q_inputs)

            if auto_weights is True:
                layer.u_weights()

            # Building and applying the neurons

            layer.apply_neurons()

            q_input = layer.get_output()
            self._q_bnn_circ.barrier() 
           
            l_counter -= 1
            self._output = q_input
            

    def train(self, auto_weights=False):
        """Trains the QBNN circuit"""
        print('\nTraining QBNN ')

        print('INPUTS: ', self.entries)
        out_count = 0
        for inp in self.entries:

            self._run_qbnn(inp, auto_weights)
            self.save_output(out_count)

            # Reverte the circuit
            self._run_qbnn(inp, auto_weights)

            out_count += 1

    def get_output(self):
        return self.outputs


def get_results(circ, output):
    """Shows the results given by the circuit"""

    print("Q_BNN results (OUTPUT): ", end='')
    circ.draw(filename='QBNN')

    clsbits = ClassicalRegister(len(output))
    circ.add_register(clsbits)
    circ.measure(output, clsbits)

    backend = Aer.get_backend('qasm_simulator')
    shots = 1
    job_sim = execute(experiments=circ, backend=backend, shots=shots)

    result = job_sim.result()

    count = result.get_counts(circ)

    qub_chance = []
    for qub, cou in zip(count.keys(), count.values()):
        chance = (cou / shots) * 100
        print("|{}> -> {:.2f}%".format(qub, chance))

        qub_chance.append((qub, chance))

    return qub_chance
