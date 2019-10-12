"""
Contains the execution of the Quantum Neural Network.
"""
from q_bnn import QBNN
from q_bnn import generate_inputs
from q_bnn import get_results
from test import QbnnTest
import numpy as np

# Running quantum circuit

def test_qbnn():
    qbnn_test = QbnnTest()
    results = {}
    for inp in generate_inputs(3):

        q_bnn = QBNN([inp], 3, 2)
        q_bnn.train(auto_weights=False)

        circuit = q_bnn.get_circuit()
        result = get_results(circuit, q_bnn.get_output())
        results[inp] = result

        try:
            qbnn_test.test_output(inp, result)
        except AssertionError:
            print('AssertionError: Unexpected result. Please verify the activation function and weights setup.')
            breakpoint()

def main():

    # Running Qbnn circuit
    test_qbnn()
    inputs = generate_inputs(3)
    q_bnn = QBNN(inputs, 3, 2)

    q_bnn.set_label(np.arange(0, 4))
    q_bnn.train(auto_weights=False)
    
    outputs = q_bnn.get_output()
    circuit = q_bnn.get_circuit()
    get_results(circuit, outputs)
    

   


if __name__ == '__main__':
    main()
