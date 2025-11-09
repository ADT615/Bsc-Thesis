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

from qiskit import *
from qiskit_algorithms.optimizers import COBYLA , SLSQP, L_BFGS_B, SPSA, NELDER_MEAD
from qiskit_algorithms import VQE
#from qiskit.utils import QuantumInstance
from qiskit.primitives import Estimator

optimizer = SLSQP(maxiter=200)

estimator = Estimator()
vqe = VQE(estimator, ansatz, optimizer)
res = vqe.compute_minimum_eigenvalue(qubit_p_op)

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

from qiskit_algorithms.eigensolvers import NumPyEigensolver

numpy_solver = NumPyEigensolver()
exact_result = numpy_solver.compute_eigenvalues(qubit_p_op)
ref_value = exact_result.eigenvalues[0]
print(f"Reference value: {ref_value  }")
print(f"VQE values: {res.optimal_value }")

# Chuyển toán tử sang dạng ma trận
from qiskit.quantum_info import SparsePauliOp
Hopt = qubit_p_op # Hamiltonian tĩnh (static)
H_static = Hopt.to_matrix()
print(Hopt)

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
T_total = 100    # Tổng thời gian tiến hóa
dt = 0.1         # Bước thời gian nhỏ
num_steps = int(T_total / dt)
num_qubits = Hopt.num_qubits  # số qubit



# Hàm tính Hamiltonian H(t)
def H_t(t, H_static, dipole_matrix, E0, Gamma):
    f_t = (E0 / np.pi) * Gamma / (Gamma**2 + t**2)
    return H_static + f_t * dipole_matrix

# Khởi tạo ma trận tiến hóa tổng U_total là ma trận đơn vị
U_total = np.eye(2**num_qubits, dtype=complex)