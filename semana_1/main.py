import qiskit as qkit
from qiskit.qasm import *
from qiskit import Aer, BasicAer
from qiskit.aqua.algorithms import DeutschJozsa
from qiskit.aqua.components.oracles import CustomCircuitOracle

# UF activate function
out_qr = qkit.QuantumRegister(1)
in_qr = qkit.QuantumRegister(5)

activate = qkit.QuantumCircuit(in_qr, out_qr)

activate.ccx(in_qr[0], in_qr[1], in_qr[3])
activate.cx(in_qr[3], in_qr[4])
activate.x(in_qr[4])
activate.barrier()

activate.ccx(in_qr[1], in_qr[2], in_qr[3])
activate.cx(in_qr[3], in_qr[4])
activate.x(in_qr[4])
activate.barrier()

activate.ccx(in_qr[0], in_qr[2], in_qr[3])
activate.cx(in_qr[3], in_qr[4])
activate.cx(in_qr[4], out_qr[0])
activate.barrier()


oracle = CustomCircuitOracle(in_qr, out_qr, out_qr[0], activate)
backend = Aer.get_backend('qasm_simulator')
print(oracle.construct_circuit())
dejo = DeutschJozsa(oracle)
result = dejo.run(backend)

print("Deutsch - Jozsa: ", result["result"])
