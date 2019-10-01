import qiskit as qkit
from qiskit.qasm import *
from qiskit import Aer, BasicAer
from qiskit.aqua.algorithms import DeutschJozsa
from qiskit.aqua.components.oracles import CustomCircuitOracle


def create_activate():
    weights = qkit.QuantumRegister(3)
    inputs = qkit.QuantumRegister(3)
    ancills = qkit.QuantumRegister(2)
    output = qkit.QuantumRegister(1)

    qbnn = qkit.QuantumCircuit(weights, inputs, ancills, output)

    # Input configuration
    #qbnn.x(inputs[0])
    qbnn.x(inputs[1])
    qbnn.x(inputs[2])

    # Applying weights
    qbnn.h(weights)

    qbnn.x(weights)

    qbnn.cx(weights[0], inputs[0])
    qbnn.cx(weights[1], inputs[1])
    qbnn.cx(weights[2], inputs[2])

    qbnn.x(weights)

    qbnn.h(weights)

    qbnn.barrier()

    # Activation function application

    qbnn.ccx(inputs[0], inputs[1], ancills[0])
    qbnn.cx(ancills[0], ancills[1])
    qbnn.x(ancills[1])
    qbnn.barrier()

    qbnn.ccx(inputs[1], inputs[2], ancills[0])
    qbnn.cx(ancills[0], ancills[1])
    qbnn.x(ancills[1])
    qbnn.barrier()

    qbnn.ccx(inputs[0], inputs[2], ancills[0])
    qbnn.cx(ancills[0], ancills[1])

    # Output storage

    qbnn.cx(ancills[1], output[0])
    qbnn.barrier()

    return qbnn, output, inputs, ancills


# UF activate function
qbnn, out_qr, v_qr, a_qr = create_activate()

cl = qkit.ClassicalRegister(1)
qbnn.add_register(cl)
qbnn.measure(out_qr, cl)
qbnn.draw(filename='QBNN')
backend = Aer.get_backend('qasm_simulator')
shots = 1024
job_sim = qkit.execute(experiments=qbnn, backend=backend, shots=1024)

count = job_sim.result().get_counts(qbnn)

print("Q_BNN results:  chance |0> : {:.2f}% and |1>: {:.2f}%".format(count['0']/shots, count['1']/shots))


"""
oracle = CustomCircuitOracle(v_qr, out_qr, a_qr, qbnn)
backend = Aer.get_backend('qasm_simulator')
print(oracle.construct_circuit())
dejo = DeutschJozsa(oracle)
result = dejo.run(backend)

print("Deutsch - Jozsa: ", result["result"])"""
