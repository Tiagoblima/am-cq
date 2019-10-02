import qiskit as qkit
from qiskit.qasm import *
from qiskit import Aer, BasicAer
from qiskit.aqua.algorithms import DeutschJozsa
from qiskit.aqua.components.oracles import CustomCircuitOracle


class QBNN:
    q_inputs = None
    q_weights = None
    q_ancillas = None
    _q_bnn_circ = qkit.QuantumCircuit()
    output = None
    s = 0

    def __init__(self):

        self._q_inputs = qkit.QuantumRegister(3)
        self._q_weights = qkit.QuantumRegister(3)
        self._q_ancillas = qkit.QuantumRegister(2)
        self.output = qkit.QuantumRegister(8)

        self._q_bnn_circ.add_register(self._q_weights, self._q_inputs, self._q_ancillas, self.output)

    # Configures the inputs
    def u_inputs(self, t_step):
        if self.s is 2:
            self.s = 0
        else:
            self.s += 1

        if t_step == 4:
            self._q_bnn_circ.x(self._q_inputs)
            self.s = 0
        else:
            self._q_bnn_circ.x(self._q_inputs[self.s])

    # Apply the weights to the inputs
    def u_weights(self):
        self._q_bnn_circ.h(self._q_weights)

        self._q_bnn_circ.x(self._q_weights)

        for i in range(3):
            self._q_bnn_circ.cx(self._q_weights[i], self._q_inputs[i])

        self._q_bnn_circ.x(self._q_weights)

        self._q_bnn_circ.h(self._q_weights)

        self._q_bnn_circ.barrier()

    # Activation function application
    def uf_activate(self):
        self._q_bnn_circ.ccx(self._q_inputs[0], self._q_inputs[1], self._q_ancillas[0])
        self._q_bnn_circ.cx(self._q_ancillas[0], self._q_ancillas[1])
        self._q_bnn_circ.x(self._q_ancillas[1])
        self._q_bnn_circ.barrier()

        self._q_bnn_circ.ccx(self._q_inputs[1], self._q_inputs[2], self._q_ancillas[0])
        self._q_bnn_circ.cx(self._q_ancillas[0], self._q_ancillas[1])
        self._q_bnn_circ.x(self._q_ancillas[1])
        self._q_bnn_circ.barrier()

        self._q_bnn_circ.ccx(self._q_inputs[0], self._q_inputs[2], self._q_ancillas[0])
        self._q_bnn_circ.cx(self._q_ancillas[0], self._q_ancillas[1])

    def save_output(self, train_step):
        # Output storage

        self._q_bnn_circ.cx(self._q_ancillas[1], self.output[train_step])
        self._q_bnn_circ.barrier()

    def train(self):
        for t_step in range(8):
            self.u_weights()
            self.uf_activate()
            self.save_output(t_step)
            self.u_inputs(t_step)

    def get_results(self):

        self._q_bnn_circ.draw(filename='QBNN')

        clsbits = qkit.ClassicalRegister(8)
        self._q_bnn_circ.add_register(clsbits)
        self._q_bnn_circ.measure(self.output, clsbits)

        backend = Aer.get_backend('qasm_simulator')
        shots = 1024
        job_sim = qkit.execute(experiments=self._q_bnn_circ, backend=backend, shots=1024)

        count = job_sim.result().get_counts(self._q_bnn_circ)

        print("Q_BNN results: ")

        for qb, c in zip(count.keys(), count.values()):
            chance = (c / shots) * 100
            print("|{}>: {:.2f}%".format(qb, chance))




