"""
Contains the class QBNN that implements the Quantum Neural Network.
"""

from itertools import combinations
from qiskit import QuantumRegister, QuantumCircuit
from qiskit import exceptions
from qiskit.quantum_info import Operator


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

    def two_activation(self, output_reg):
        """The implementation of u_activate for two qubits"""
        self._q_neuron.ccx(self.inputs[0], self.inputs[1], self._output[output_reg])
        self._q_neuron.cx(self.inputs[0], self._output[output_reg])
        self._q_neuron.cx(self.inputs[1], self._output[output_reg])

    def three_activation(self, output_reg):
        """The implementation of u_activate for three qubits"""
        self._q_neuron.ccx(self.inputs[0], self.inputs[1], self._output[output_reg])
        self._q_neuron.ccx(self.inputs[1], self.inputs[2], self._output[output_reg])
        self._q_neuron.ccx(self.inputs[0], self.inputs[2], self._output[output_reg])

    def large_activation(self, output_reg):
        """The implementation of u_activate for more than 3 qubits"""
        inps = list(combinations(self.inputs, 1))
        for inp in inps:
            self._q_neuron.x(inp[0])
            self._q_neuron.mct(self.inputs, self._output[output_reg], self._ancillas)
            self._q_neuron.x(inp[0])
        self._q_neuron.mct(self.inputs, self._output[output_reg], self._ancillas)

    def set_ouput(self, q_ouput):
        """Adds the output qubit if it necessary"""
        self.n_outputs = len(q_ouput) # It is also the quantity of required neurons 
        self._output = q_ouput
        try: 
            self._q_neuron.add_register(self._output)
        except exceptions.QiskitError:
            pass

    def set_ancillas(self, q_ancillas):
        self._ancillas = q_ancillas

    def set_output_reg(self, output_reg):
        self._output = output_reg

    def get_circuit(self):
        """Returns the Q_neuron circuit"""
        return self._q_neuron

    def set_weights(self, entry=None):
        """Sets the weights manually by receiving the position that might be changed to 1"""
        if entry is None:
            entry = []
        for pos in entry:
            self._q_neuron.x(self._weights[int(pos)])

    def u_inputs(self, entry=None):
        """Configures the inputs setting to 1 the inputs in the position 'pos' """
        if entry is None:
            entry = []
        for pos in entry:
            self._q_neuron.x(self.inputs[int(pos)])

    def init_weights(self):
        """Initialize the weights in superposition"""
        self._q_neuron.h(self._weights) 
        self._q_neuron.x(self._weights)
        
    def u_weights(self):
        """Apply the weights to the inputs"""
        for i in range(self.n_inputs):
            self._q_neuron.cx(self._weights[i], self.inputs[i])
            
    def set_input_rg(self, regs):
        self.inputs = regs
        self._q_neuron.add_register(regs)

    def apply_neurons(self):
        """Iterates through the number of neurons and applies 
        the u_activate module as much as necessary"""
        for neuron in range(self.n_outputs):
            self.uf_activate(neuron)

    def uf_activate(self, output_reg):
        """Applies the activation function"""
        if len(self.inputs) is 2:
            self.two_activation(output_reg)
        elif len(self.inputs) is 3:
            self.three_activation(output_reg)
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
    entries = []
    n_inputs = 0
    labels_set = []
    inputs = None
    _output = None 
    l_outs = []
    _labels = None


    def __init__(self, entries, input_size, n_layers, add_q_output=True, gene_lout=True):

        self.entries = entries
        self.n_inputs = len(entries)
        self.n_layers = n_layers
        self.outputs = QuantumRegister(self.n_inputs, name='out')
        self.inputs = QuantumRegister(input_size, name='i')
        self.weights = QuantumRegister(input_size, name='w')
        self._q_bnn_circ = QuantumCircuit(self.weights, self.inputs)
        
        if add_q_output:
            self._q_bnn_circ.add_register(self.outputs)
        
        self.l_outs = []

        if gene_lout:
            for l_layer in range(n_layers, 0, -1):
                self.l_outs.append(QuantumRegister(l_layer, name='l'+str(n_layers-l_layer)+'_out'))

    def set_label(self, labels_set=None):
        """Configures the expected label for each input"""
        for pos in labels_set:
            self._q_bnn_circ.x(self.outputs[int(pos)])

    def get_circuit(self):
        """Returns the QBNN circuit"""
        return self._q_bnn_circ

    def get_output(self):
        return self.outputs

    def get_n_layers(self):
        return self.n_layers
    
    def get_layers_output(self):
        return self.l_outs

    def set_layers_output(self, q_outputs=[]):
        self.l_outs = q_outputs
    
    def get_q_input(self):
        return self.inputs

    def set_q_input(self, q_inputs):
        self.q_inputs = q_inputs

    def get_q_weights(self):
        return self.weights
        
    def save_output(self, output_reg):
        """Saves the output in a specific qubit for each input"""
        self._q_bnn_circ.cx(self._output, self.outputs[output_reg])

    def set_q_outputs(self, q_ouputs):
        try:
           self._q_bnn_circ.add_register(q_ouputs)
           self.outputs = q_ouputs
        except exceptions.QiskitError:
            self.outputs = q_ouputs
            pass

    def reverse_init_sign(self):
        """Reverse the sign of the initial state"""
        for inp, weight in zip(self.inputs, self.weights):
            self._q_bnn_circ.z(inp)
            self._q_bnn_circ.z(weight)
    
    def create_qbnn(self, weights, inputs=None, init_weights=True, reverse=False):
        """Applies the gates of the q_bnn circuit"""
        if weights is None:
            weights = [[]]

        q_input = self.inputs
        l_counter = self.n_layers
        apply_weight = True
        apply_inputs = True

        for l_out in self.l_outs:
            
            layer = Layer(self._q_bnn_circ, self.weights, q_input)
            layer.set_ouput(l_out)

            if apply_inputs:
                layer.u_inputs(inputs)
            if init_weights and apply_weight:
                layer.init_weights()
                layer.u_weights()
            elif apply_weight:
                layer.set_weights(weights[self.n_layers-l_counter])
                layer.u_weights()
                layer.set_weights(weights[self.n_layers-l_counter])

            # Building and applying the neurons
            layer.apply_neurons()
            q_input = layer.get_output()   
            apply_weight = False
            apply_inputs = False
            l_counter -= 1
        self._output = q_input
              
    def get_operator(self):
       return Operator(self._q_bnn_circ)

    def feed_qbnn(self, init_weights=False, weights_list=None):
        """Feeds the QBNN circuit with the entries"""
        
        print('\nFeeding QBNN ')
        if weights_list is None:
            weights_list = [[] for elem in range(self.n_layers)]
           
        print('INPUTS: ', self.entries)
        out_count = 0
        for inp in self.entries:
            self.create_qbnn(weights_list, inp , init_weights)
            self.save_output(out_count)
            out_count += 1



