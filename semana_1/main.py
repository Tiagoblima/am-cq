"""
Contains the execution of the Quantum Neural Network.
"""
from q_bnn import QBNN


# Running quantum circuit

Q_BNN = QBNN()

REPLY = input('The following algorithm consumes a lot of CPU processing continue? [y/n]\n')
Y = 'y'
if REPLY.lower() is Y:
    Q_BNN.train()
    Q_BNN.get_results()
else:
    print('End of program')
