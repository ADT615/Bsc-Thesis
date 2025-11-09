from qiskit.circuit import Parameter, QuantumCircuit, ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli
import numpy as np

def create_parameterized_hamiltonian_ansatz(nqubits, pauli_labels_fixed, num_layers, initial_coeffs_values=None, add_classical_bits=True):
    """
    Tạo một Hamiltonian ansatz có tham số U(theta) = product_L (product_k exp(-i * theta_Lk * P_k)).

    Args:
        nqubits (int): Số qubit.
        pauli_labels_fixed (list[str]): Danh sách các chuỗi Pauli CỐ ĐỊNH (P_k).
        num_layers (int): Số lớp (N_L) của ansatz.
        initial_coeffs_values (list[float] or np.ndarray, optional): 
                               Giá trị khởi tạo cho các tham số theta. 
                               Nếu num_layers > 1, đây có thể là một mảng 1D dẹt (flattened)
                               hoặc bạn có thể cấu trúc nó theo lớp. 
                               Độ dài phải là len(pauli_labels_fixed) * num_layers.
        add_classical_bits (bool): Nếu True, thêm bit cổ điển.

    Returns:
        qiskit.QuantumCircuit: Mạch lượng tử ansatz có tham số.
        list[Parameter]: Danh sách tất cả các tham số đã tạo (dạng dẹt).
    """
    if not pauli_labels_fixed:
        raise ValueError("pauli_labels_fixed không được rỗng.")

    if add_classical_bits:
        circuit = QuantumCircuit(nqubits, nqubits)
    else:
        circuit = QuantumCircuit(nqubits)

    total_parameters = len(pauli_labels_fixed) * num_layers
    params_vector = ParameterVector('θ', total_parameters)
    
    all_parameters_list = list(params_vector) # Danh sách các đối tượng Parameter

    param_idx_counter = 0
    for _ in range(num_layers): # Lặp qua các lớp
        for pauli_str in pauli_labels_fixed: # Lặp qua các toán tử Pauli cố định
            if not pauli_str or len(pauli_str) != nqubits:
                raise ValueError(f"Chuỗi Pauli '{pauli_str}' không hợp lệ.")
            
            current_pauli_operator = Pauli(pauli_str)
            # time trong PauliEvolutionGate bây giờ là một Parameter
            param_coeff = params_vector[param_idx_counter]
            param_idx_counter += 1
            
            evolution_gate = PauliEvolutionGate(current_pauli_operator, time=param_coeff) 
            circuit.append(evolution_gate, range(nqubits))
        if num_layers > 1 : # Thêm barrier giữa các lớp nếu có nhiều lớp
            circuit.barrier()
            
    # Cung cấp giá trị khởi tạo nếu có
    # Qiskit sẽ tự động xử lý việc gán giá trị ban đầu này khi dùng trong optimizer
    # Hoặc có thể gán chúng khi chạy optimizer
    # Hàm này chỉ trả về mạch và các tham số.
    
    return circuit, all_parameters_list

# 1. Lấy các chuỗi Pauli CỐ ĐỊNH từ Hamiltonian 
# Đây sẽ là CƠ SỞ P_k cho ansatz 
# Có thể lấy chúng từ H_int(T) , hoặc trực tiếp từ H(t)
# nếu các P_k không thay đổi theo thời gian (chỉ có hệ số lambda_j(t) thay đổi).
# Dựa trên H_total1 = Hopt + f(t)*dipole_qubit, các P_k là cố định.

# Lấy labels từ Hopt (phần tĩnh)
static_pauli_op = Hopt 
static_labels = static_pauli_op.paulis.to_labels()

# Lấy labels từ dipole_qubit (phần tương tác)
dipole_pauli_op = dipole_qubit
dipole_interaction_labels = dipole_pauli_op.paulis.to_labels()

# Kết hợp danh sách labels và loại bỏ trùng lặp. QUAN TRỌNG: Giữ thứ tự nhất quán.
# Cách tốt nhất là tạo một SparsePauliOp tổng H(t) = H_opt + f(t)*dipole_qubit rồi lấy paulis từ đó.
#  Tuy nhiên, f(t) là số, nên paulis của H(t) sẽ giống paulis của (H_opt + dipole_qubit) nếu không có sự triệt tiêu.
# Để đơn giản, ta có thể lấy một bộ cơ sở Pauli từ H_opt và dipole_qubit.
# Tạo một bộ cơ sở các Pauli strings duy nhất từ cả Hopt và dipole_qubit
# Cách này đảm bảo không trùng lặp và có thứ tự nhất quán nếu dùng dict.fromkeys

combined_unique_labels = list(dict.fromkeys(static_labels + dipole_interaction_labels))
num_qubits = Hopt.num_qubits

# 2. Tạo Ansatz U(theta)
N = num_qubits # num_qubits = 4
num_ansatz_layers = 2 # Số lớp cho ansatz, thử nghiệm

# Giá trị khởi tạo cho các tham số theta (tùy chọn, nhưng có thể hữu ích)
# Ví dụ: bạn có thể dùng các coeffs đã tính từ time_dependent_integral làm điểm khởi đầu cho lớp ĐẦU TIÊN nếu num_layers = 1.
# Nếu dùng nhiều lớp, việc khởi tạo cần cẩn thận hơn.

# integral_H_op_at_T = time_dependent_integral(H_time, t=T) # T=10
# initial_coeffs_for_ansatz = []
# label_to_coeff_map = dict(zip(integral_H_op_at_T.paulis.to_labels(), integral_H_op_at_T.coeffs.real))
# for label in combined_unique_labels:
#     initial_coeffs_for_ansatz.append(label_to_coeff_map.get(label, 0.0)) # Gán 0 nếu label không có trong H_int

# Tạo ansatz và danh sách các tham số có thể tối ưu
# Bây giờ, 'combined_unique_labels' sẽ là P_k cố định của bạn
# và các 'theta_k' sẽ được tạo bên trong hàm này
ansatz_u, optimizable_parameters = create_parameterized_hamiltonian_ansatz(
    nqubits=N,
    pauli_labels_fixed=combined_unique_labels,
    num_layers=num_ansatz_layers,
    add_classical_bits=True # Giữ True để nhất quán với target_state nếu nó có clbits
    # initial_coeffs_values có thể được truyền vào optimizer sau
)

print("Ansatz U(theta) được tạo:")
print(ansatz_u.draw(output='text'))
print(f"Số lượng tham số có thể tối ưu: {len(optimizable_parameters)}")
print(f"Các tham số: {optimizable_parameters}")

from qoop.compilation.qsp import QuantumStatePreparation
p0s = []
times = np.linspace(0, 10, 4) # Thời gian cho vòng lặp QSP

for time in times:
    target_unitary_circuit = time_dependent(N, H_time, time) # time_dependent từ cell [63]
    target_s_inverse = target_unitary_circuit.inverse()

    # Giá trị khởi tạo cho các tham số cho lần fit này (quan trọng!)
    # Nếu không có, Qiskit/qoop có thể dùng giá trị ngẫu nhiên hoặc 0.
    # Bạn có thể tính H_int(time_val) và dùng coeffs của nó làm điểm khởi đầu cho các theta.
    current_H_int = time_dependent_integral(H_time, t=time)
    initial_point_for_fit = []
    label_to_coeff_map_current_t = dict(zip(current_H_int.paulis.to_labels(), current_H_int.coeffs.real))
    for _ in range(num_ansatz_layers): # Nếu có nhiều lớp, lặp lại bộ giá trị khởi tạo
        for label in combined_unique_labels:
            initial_point_for_fit.append(label_to_coeff_map_current_t.get(label, np.random.rand())) # Lấy coeff hoặc ngẫu nhiên


    qsp = QuantumStatePreparation(
        u=ansatz_u, # Ansatz CÓ THAM SỐ
        target_state=target_s_inverse
    )
    
    # Hàm fit của qoop cần có khả năng nhận optimizer và initial_point
    # API của qoop.QuantumStatePreparation.fit() cần được kiểm tra để biết
    # cách truyền optimizer và điểm khởi đầu cho các tham số của 'u'.
    # Giả sử nó tự dùng một optimizer mặc định và khởi tạo ngẫu nhiên nếu không có initial_point.
    # Hoặc bạn cần một vòng lặp tối ưu hóa bên ngoài như đã thảo luận.
    
    # Dựa trên mô tả qsp.ipynb, .fit() sẽ tối ưu các tham số trong ansatz_u
    result_fit = qsp.fit(num_steps=300, # Hoặc nhiều hơn
                         metrics_func=['loss_basic'],
                         initial_point=np.array(initial_point_for_fit) # Cung cấp điểm khởi đầu
                        ) 
    # Sau khi fit, ansatz_u.parameters sẽ được gán các giá trị tối ưu (nếu fit sửa đổi inplace)
    # hoặc result_fit.optimal_parameters sẽ chứa chúng. 
    # qsp.compiler.metrics sẽ chứa loss.

    if 'loss_basic' in result_fit.compiler.metrics:
        p0s.append(1 - result_fit.compiler.metrics['loss_basic'][-1])
    else:
        print(f"Không tìm thấy 'loss_basic' cho time {time}. Keys: {result_fit.compiler.metrics.keys()}")
        p0s.append(0) # Hoặc một giá trị lỗi

print('Mean loss (với ansatz có tham số)', p0s)