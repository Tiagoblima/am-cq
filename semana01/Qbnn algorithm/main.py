"""
Contains the execution of the Quantum Neural Network.
"""
from qiskit import Aer
from q_bnn import QBNN
from test import QbnnTest
from util.util import generate_positions, get_results

import numpy as np


def test_qbnn():
    """Test the qbnn using the test module"""

    qbnn_test = QbnnTest()
    results = {}
    for inp in generate_positions(3):

        q_bnn = QBNN([inp], 3, 2)
        q_bnn.feed_qbnn(init_weights=False)

        circuit = q_bnn.get_circuit()
        result = get_results(circuit, q_bnn.get_output())
        results[inp] = result

        try:
            qbnn_test.test_output(inp, result)
        except AssertionError:
            print('AssertionError: Unexpected result. Please verify the activation function and weights setup.')
            breakpoint()


def main():
    test_qbnn()
    inputs = generate_positions(3)
    q_bnn = QBNN(inputs, 3, 2)
    q_bnn.set_label(np.arange(0, 4))
    q_bnn.feed_qbnn(init_weights=True, weights_list=None)

    circuit = q_bnn.get_circuit()
    get_results(circuit, q_bnn.get_output())


if __name__ == '__main__':
    main()
