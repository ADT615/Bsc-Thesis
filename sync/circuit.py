import qiskit
from qiskit.quantum_info import Operator
import numpy as np
import pennylane as qml
def mcx_gate(num_qubits):
    """_summary_

    Args:
        num_qubits (_type_): _description_

    Returns:
        _type_: _description_
    """
    mcx = qiskit.circuit.library.MCXGate(num_qubits - 1)
    return np.real(Operator(mcx).data)

# def u3cry(num_qubits: int, num_layers: int, thetas: qiskit.circuit.ParameterVector, active_blocks: np.ndarray) -> tuple[qiskit.QuantumCircuit, np.ndarray]:
#     """_summary_

#     Args:
#         num_qubits (int): _description_
#         num_layers (int): _description_
#         thetas (qiskit.circuit.ParameterVector): _description_
#         active_blocks (np.ndarray): _description_

#     Returns:
#         tuple[qiskit.QuantumCircuit, np.ndarray]: _description_
#     """
#     qc = qiskit.QuantumCircuit(num_qubits)
#     k = 0
#     index_params = 0
#     for _ in range(num_layers):
#         for i in range(num_qubits):
#             for j in range(i + 1, num_qubits):
#                 if active_blocks[k] == 1:
#                     qc.u(thetas[index_params],thetas[index_params + 1],thetas[index_params + 2],i)
#                     qc.u(thetas[index_params + 3],thetas[index_params + 4],thetas[index_params + 5],j)
#                     qc.cx(j, i)
#                     index_params += 6
#                 k += 1
#                 qc.barrier()
#     return qiskit.quantum_info.Operator(qc).data # .draw('mpl')

# def u3cry_qc(num_qubits: int, num_layers: int, thetas: qiskit.circuit.ParameterVector, active_blocks: np.ndarray) -> tuple[qiskit.QuantumCircuit, np.ndarray]:
#     """_summary_

#     Args:
#         num_qubits (int): _description_
#         num_layers (int): _description_
#         thetas (qiskit.circuit.ParameterVector): _description_
#         active_blocks (np.ndarray): _description_

#     Returns:
#         tuple[qiskit.QuantumCircuit, np.ndarray]: _description_
#     """
#     qc = qiskit.QuantumCircuit(num_qubits)
#     k = 0
#     index_params = 0
#     for _ in range(num_layers):
#         for i in range(num_qubits):
#             for j in range(i + 1, num_qubits):
#                 if active_blocks[k] == 1:
#                     qc.u(thetas[index_params],thetas[index_params + 1],thetas[index_params + 2],i)
#                     qc.u(thetas[index_params + 3],thetas[index_params + 4],thetas[index_params + 5],j)
#                     qc.cx(j, i)
#                     index_params += 6
#                 k += 1
#                 qc.barrier()
#     return qc



def u3cry_wrapper(num_qubits: int, num_layers: int, active_blocks: np.ndarray):
    dev = qml.device('default.qubit', wires=num_qubits)
    @qml.qnode(dev, diff_method="parameter-shift")
    def u3cry(thetas):
        k = 0
        index_params = 0
        for _ in range(num_layers):
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    if active_blocks[k] == 1:
                        qml.U3(thetas[index_params],thetas[index_params + 1],thetas[index_params + 2],i)
                        qml.U3(thetas[index_params + 3],thetas[index_params + 4],thetas[index_params + 5],j)
                        qml.CNOT([j, i])
                        qml.Barrier()
                        index_params += 6
                    k += 1
    return u3cry