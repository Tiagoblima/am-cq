"""
Contains the execution of the Quantum Neural Network.
"""
from q_bnn import QBNN
from q_bnn import generate_inputs
from q_bnn import get_results
from test import QbnnTest


# Running quantum circuit

def main():
    qbnn_test = QbnnTest()
    results = {}
    for inp in generate_inputs(3):

        q_bnn = QBNN([inp], 3)
        q_bnn.train(auto_weights=False)

        circuit = q_bnn.get_circuit()
        result = get_results(circuit, q_bnn.get_output())
        results[inp] = result

        try:
            qbnn_test.test_output(inp, result)
        except AssertionError:
            print('AssertionError: Unexpected result. Please verify the activation function and weights setup.')
            breakpoint()


if __name__ == '__main__':
    main()
