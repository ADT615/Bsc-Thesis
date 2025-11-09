
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit

driver = PySCFDriver(
    atom="H 0 0 0; H 0 0 0.735",
    basis="sto3g",
    charge=0,
    spin=0,
    unit=DistanceUnit.ANGSTROM,
)
problem = driver.run()
hamiltonian = problem.hamiltonian

# Thêm moment lưỡng cực vào Hamiltonian

import numpy as np 
from qiskit_nature.second_q.operators import PolynomialTensor
from qiskit_nature.second_q.properties import ElectronicDipoleMoment


dipole: ElectronicDipoleMoment = problem.properties.electronic_dipole_moment

if dipole is not None:
    nuclear_dip = dipole.nuclear_dipole_moment
    # Cập nhật moment lưỡng cực cho các thành phần x, y, z
    dipole.x_dipole.alpha += PolynomialTensor({"": nuclear_dip[0]})
    dipole.y_dipole.alpha += PolynomialTensor({"": nuclear_dip[1]})
    dipole.z_dipole.alpha += PolynomialTensor({"": nuclear_dip[2]})
    print("Đã thêm moment lưỡng cực vào Hamiltonian.")
else:
    print("Moment lưỡng cực không tồn tại trong problem.")

# Kiểm tra coefficients của Hamiltonian
coefficients = hamiltonian.electronic_integrals
print("Coefficients của Hamiltonian:", coefficients.alpha)

second_q_op = hamiltonian.second_q_op()
print(second_q_op)

from qiskit_nature.second_q.mappers import JordanWignerMapper

mapper = JordanWignerMapper()
qubit_p_op = mapper.map(second_q_op)
print(qubit_p_op)

#

from qiskit_nature.second_q.circuit.library import HartreeFock, UCC

ansatz = UCC(
    num_spatial_orbitals=2,
    num_particles=[1,1],
    excitations='sd',
    qubit_mapper=mapper,
    initial_state=HartreeFock(
        num_spatial_orbitals=2,
        num_particles=[1,1],
        qubit_mapper=mapper,
    ),
    reps=1,

)

#

from qiskit import *
from qiskit_algorithms.optimizers import COBYLA , SLSQP, L_BFGS_B, SPSA, NELDER_MEAD
from qiskit_algorithms import VQE
#from qiskit.utils import QuantumInstance
from qiskit.primitives import Estimator

optimizer = SLSQP(maxiter=200)

estimator = Estimator()
vqe = VQE(estimator, ansatz, optimizer)
res = vqe.compute_minimum_eigenvalue(qubit_p_op)

#


import numpy as np


# we will iterate over these different optimizers
optimizers = [COBYLA(maxiter=80), L_BFGS_B(maxiter=60), SLSQP(maxiter=60)]
converge_counts = np.empty([len(optimizers)], dtype=object)
converge_vals = np.empty([len(optimizers)], dtype=object)

for i, optimizer in enumerate(optimizers):
    print("\rOptimizer: {}        ".format(type(optimizer).__name__), end="")
    #algorithm_globals.random_seed = 50
    #ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")

    counts = []
    values = []

    def store_intermediate_result(eval_count, parameters, mean, std):
        counts.append(eval_count)
        values.append(mean)

    vqe = VQE(estimator, ansatz, optimizer, callback=store_intermediate_result)
    res = vqe.compute_minimum_eigenvalue(qubit_p_op)
    converge_counts[i] = np.asarray(counts)
    converge_vals[i] = np.asarray(values)

#

from qiskit_algorithms.eigensolvers import NumPyEigensolver

numpy_solver = NumPyEigensolver()
exact_result = numpy_solver.compute_eigenvalues(qubit_p_op)
ref_value = exact_result.eigenvalues[0]
print(f"Reference value: {ref_value  }")
print(f"VQE values: {res.optimal_value }")

# Chuyển toán tử sang dạng ma trận
from qiskit.quantum_info import SparsePauliOp
Hopt = SparsePauliOp(['IIII', 'IIIZ', 'IIZI', 'IZII', 'ZIII', 'IIZZ', 'IZIZ', 'ZIIZ', 'YYYY', 'XXYY', 'YYXX', 'XXXX', 'IZZI', 'ZIZI', 'ZZII'],
              coeffs=[-0.81054798+0.j,  0.17218393+0.j, -0.22575349+0.j,  0.17218393+0.j,
 -0.22575349+0.j,  0.12091263+0.j,  0.16892754+0.j,  0.16614543+0.j,
  0.0452328 +0.j,  0.0452328 +0.j,  0.0452328 +0.j,  0.0452328 +0.j,
  0.16614543+0.j,  0.17464343+0.j,  0.12091263+0.j]) # Hamiltonian tĩnh (static)
H_static = Hopt.to_matrix()

dipole_ops = dipole.second_q_ops()
print("Nội dung của dipole_ops:", dipole_ops)
# Lấy toán tử moment lưỡng cực từ phương Z (vì X và Y rỗng)
dipole_op = dipole_ops["ZDipole"]
dipole_qubit = mapper.map(dipole_op)
dipole_matrix = dipole_qubit.to_matrix()

# Thiết lập các tham số cho trường ngoài phụ thuộc thời gian
Gamma = 0.25
E0 = 0.01
# Thiết lập các tham số tiến hóa thời gian
T_total = 1000    # Tổng thời gian tiến hóa
dt = 0.1         # Bước thời gian nhỏ
num_steps = int(T_total / dt)
num_qubits = Hopt.num_qubits  # số qubit

# Khởi tạo ma trận tiến hóa tổng U_total là ma trận đơn vị
U_total = np.eye(2**num_qubits, dtype=complex)

# Tạo danh sách các thời điểm từ 0 đến T_total với bước dt
time_points = np.arange(0, T_total, dt)

# Tính toán ma trận tiến hóa cho từng bước thời gian

from scipy.linalg import expm

# Vòng lặp tiến hóa theo thời gian: tại mỗi bước tính toán Hamiltonian H(t)
for t_current in time_points:
    # Hàm điều chế trường ngoài: f(t) = (E0/π)*Gamma/(Gamma² + t²)
    f_t = (E0 / np.pi) * Gamma / (Gamma**2 + t_current**2)
    # Hamiltonian tại thời điểm t_current: H(t) = H_static + f(t)*dipole_matrix
    H_t = H_static + f_t * dipole_matrix
    # Tính ma trận tiến hóa cho bước dt: U_step = exp(-i H(t) dt)
    U_step = expm(-1j * dt * H_t)
    # Cập nhật tiến hóa tổng theo thứ tự thời gian
    U_total = U_step @ U_total

import qiskit 
#  Tạo mạch lượng tử từ ma trận tiến hóa U_total
from qiskit.circuit.library import UnitaryGate

qc = qiskit.QuantumCircuit(num_qubits)
unitary_gate = UnitaryGate(U_total)
qc.append(unitary_gate, range(num_qubits))

print("Mạch tiến hóa thời gian:")
print(qc.draw(output='text'))

from qoop.compilation.qsp import QuantumStatePreparation
compiler = QuantumStatePreparation.prepare(U_total)
print("Các tham số sau khi biên dịch:")
print(compiler.thetas)

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from qiskit.quantum_info import Statevector



num_qubits = Hopt.num_qubits
#psi_0 = Statevector.from_label('0' * num_qubits)
#psi_0 = Statevector.from_label('0001')  # Thay đổi trạng thái ban đầu
qc = QuantumCircuit(num_qubits)
qc.append(ansatz.assign_parameters(res.optimal_parameters), range(num_qubits))

psi_0 = Statevector(qc)

T_total = 1000  # Tổng thời gian mô phỏng
dt = 0.1       # Bước thời gian
num_steps = int(T_total / dt)
time_points = np.arange(0, T_total, dt)

mu_t = []  # Danh sách moment lưỡng cực theo thời gian
psi_t = psi_0.data  # Trạng thái ban đầu
U_t = np.eye(2**num_qubits, dtype=complex)  # Ma trận tiến hóa

for t_current in time_points:
    f_t = (E0 / np.pi) * Gamma / (Gamma**2 + t_current**2)  # Hệ số trường ngoài
    H_t = H_static + f_t * dipole_matrix  # Hamiltonian tại thời điểm t
    U_step = expm(-1j * dt * H_t)  # Ma trận tiến hóa bước dt
    U_t = U_step @ U_t  # Cập nhật tiến hóa tổng
    psi_t = U_t @ psi_0.data  # Tiến hóa trạng thái

    # Moment lưỡng cực tại thời điểm t
    mu_t.append(np.real(psi_t.conj().T @ dipole_matrix @ psi_t))


plt.figure(figsize=(8, 5))
plt.plot(time_points, mu_t, 'b', lw=2, label="Moment lưỡng cực ⟨μ_z(t)⟩")
plt.xlabel("Thời gian (t)")
plt.ylabel("Moment lưỡng cực ⟨μ_z⟩")
plt.title("Tiến hóa moment lưỡng cực theo thời gian")
plt.legend()
plt.grid()
plt.show()
