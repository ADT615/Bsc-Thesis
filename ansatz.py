"""
Ansatz module
Chứa các hàm định nghĩa quantum circuit ansatz cho PennyLane
"""

import pennylane as qml
import pennylane.numpy as np


def pennylane_ansatz_from_qiskit_pauli_evo(thetas, num_qubits, pauli_labels, num_layers):
    """
    Tạo một ansatz Pennylane tương đương với việc áp dụng một chuỗi các PauliEvolutionGate của Qiskit.
    Cấu trúc của Qiskit ansatz: U(theta) = product_L (product_k exp(-i * theta_Lk * P_k))
    Trong Pennylane, qml.exp(op, coeff) thực hiện exp(coeff * op).
    Chúng ta muốn thực hiện exp(-i * theta_Lk * P_k), vậy:
        op = P_k (dưới dạng PauliWord của Pennylane)
        coeff = -1j * theta_Lk

    Args:
        thetas (np.ndarray): Mảng 1D các tham số theta, được làm phẳng (flattened).
                                Thứ tự phải khớp với cách Qiskit ansatz sử dụng chúng
                                (tức là, tất cả các theta cho lớp 1, sau đó tất cả cho lớp 2, v.v.).
        num_qubits (int): Số qubit của hệ thống.
        pauli_labels (list[str]): Danh sách các chuỗi Pauli CỐ ĐỊNH (P_k)
                                        dưới dạng chuỗi ký tự kiểu Qiskit (ví dụ: "IXYZ", "ZZI").
                                        Độ dài mỗi chuỗi phải bằng num_qubits.
        num_layers (int): Số lớp (N_L) của ansatz.
    """
    if not pauli_labels:
        raise ValueError("pauli_labels must not be empty.")

    num_params_per_layer = len(pauli_labels)
    num_params_expected = num_params_per_layer * num_layers

    if len(thetas) != num_params_expected:
         raise ValueError(f"Expected {num_params_expected} parameters, got {len(thetas)}.")

    param_idx = 0
    for _ in range(num_layers):
        for pauli_qiskit_str in pauli_labels:
            if len(pauli_qiskit_str) != num_qubits:
                raise ValueError(f"Pauli string '{pauli_qiskit_str}' does not match num_qubits={num_qubits}.")
            
            pennylane_observable = None
            if all(c == 'I' for c in pauli_qiskit_str):
                if num_qubits > 0:
                    pennylane_observable = qml.Identity(0)
                    for i_w in range(1, num_qubits): 
                        pennylane_observable @= qml.Identity(i_w)
                elif num_qubits == 0: 
                    pennylane_observable = qml.Identity(0)
                else: 
                    raise ValueError("Số qubit phải không âm.")
            else:
                try: 
                    pennylane_observable = qml.pauli.string_to_pauli_word(pauli_qiskit_str)
                except AttributeError:
                    ops_list = []
                    for i_wire in range(num_qubits):
                        char = pauli_qiskit_str[i_wire]
                        if char == 'X': 
                            ops_list.append(qml.PauliX(i_wire))
                        elif char == 'Y': 
                            ops_list.append(qml.PauliY(i_wire))
                        elif char == 'Z': 
                            ops_list.append(qml.PauliZ(i_wire))
                        elif char == 'I': 
                            pass
                        else: 
                            raise ValueError(f"Ký tự Pauli không hợp lệ '{char}'")

                    if not ops_list and num_qubits > 0:
                         pennylane_observable = qml.Identity(0)
                         for i_w in range(1, num_qubits): 
                             pennylane_observable @= qml.Identity(i_w)
                    elif ops_list: 
                        pennylane_observable = qml.prod(*ops_list)
                    else: 
                        raise ValueError(f"Không tạo được observable từ '{pauli_qiskit_str}'")
            
            if pennylane_observable is not None: 
                qml.exp(pennylane_observable, -1j * thetas[param_idx])
            param_idx += 1


def create_pauli_labels_from_operators(static_pauli_op, dipole_pauli_op):
    """
    Tạo danh sách các Pauli labels từ static và dipole operators
    
    Args:
        static_pauli_op: Static Hamiltonian Pauli operator
        dipole_pauli_op: Dipole Pauli operator
        
    Returns:
        list: Combined unique Pauli labels
    """
    static_labels = static_pauli_op.paulis.to_labels()
    dipole_interaction_labels = dipole_pauli_op.paulis.to_labels()
    combined_unique_labels = list(dict.fromkeys(static_labels + dipole_interaction_labels))
    return combined_unique_labels
