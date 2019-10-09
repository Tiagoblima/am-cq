"""
Contains the execution of the Quantum Neural Network.
"""
from q_bnn import QBNN
from q_bnn import generate_inputs
from q_bnn import get_results

# Running quantum circuit
results = {}
for inp in generate_inputs(3):
    results[inp] = []
    Q_BNN = QBNN([inp], 3)
    Q_BNN.train()

    circuit = Q_BNN.get_circuit()
    get_results(results, circuit, Q_BNN.get_output(), inp)

print(results)


