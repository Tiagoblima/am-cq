"""
Contains the execution of the Quantum Neural Network.
"""
from q_bnn import QBNN
from q_bnn import generate_inputs
from q_bnn import get_results

# Running quantum circuit

Q_BNN = QBNN(generate_inputs(3), 3)
Q_BNN.train()

get_results(Q_BNN.get_circuit(), 1, Q_BNN.get_output())
#REPLY = input('The following algorithm consumes a lot of CPU processing continue? [y/n]\n')
Y = 'y'


#Q_BNN.get_results()

