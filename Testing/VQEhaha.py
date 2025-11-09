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