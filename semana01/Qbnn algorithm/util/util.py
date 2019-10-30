from itertools import combinations
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute
import numpy as np


def generate_positions(inputs_size):
    """Generate the inputs to the QBNN combining the positions of 
       the quantum registers that can be changed to 1"""

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


def get_results(circ, output, backend=None, shots=1, circ_name='CIRCUIT'):
    """Shows the results given by the circuit"""

    if backend is None:
        backend = Aer.get_backend('qasm_simulator')

    print("OUTPUT: ", end='')
    circ.draw(filename=circ_name)

    clsbits = ClassicalRegister(len(output))
    circ.add_register(clsbits)
    circ.measure(output, clsbits)

    job_sim = execute(experiments=circ, backend=backend, shots=shots)

    result = job_sim.result()

    count = result.get_counts(circ)

    qub_chance = []
    for qub, cou in zip(count.keys(), count.values()):
        chance = (cou / shots) * 100
        print("|{}> -> {:.2f}%".format(qub, chance))

        qub_chance.append((qub, chance))

    return qub_chance
