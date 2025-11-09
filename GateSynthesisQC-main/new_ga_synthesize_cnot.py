import qiskit
from qoop.core import state, metric, ansatz
from qoop.backend import constant
from qoop.compilation.qsp import QuantumStatePreparation
from qoop.evolution.environment_synthesis import MetadataSynthesis
from qoop.evolution.environment import EEnvironment
from qoop.evolution import crossover, divider, normalizer, generator, threshold, mutate
from qiskit.quantum_info import random_unitary
from qiskit.circuit.library import UnitaryGate
import pennylane as qml
from sync import cost as cost_func
import pennylane.numpy as nps
import pennylane as qml
target = UnitaryGate(random_unitary(2**3))
def fitness_synthesis(qc_qiskit: qiskit.QuantumCircuit):
    dev = qml.device("default.qubit")
    qfunc = qml.from_qiskit(qc_qiskit, measurements=qml.expval(qml.Z(0)))
    qc = qml.QNode(qfunc, dev)

    thetas = nps.random.uniform(0, 2*nps.pi, len(qc_qiskit.parameters))
    def cost(thetas):
        matrix = qml.matrix(qc, wire_order=list(range(qc_qiskit.num_qubits)))(thetas)
        return cost_func.c_hst(matrix, target)

    steps = 1000
    costs = []
    opt = qml.AdamOptimizer(stepsize = 0.01)
    for n in range(steps):
        thetas, prev_cost = opt.step_and_cost(cost, thetas)
        if prev_cost < 10**(-4):
            print("Achieved error threshold at step", n)
            break
        costs.append(prev_cost)
    return 1-costs[-1].numpy()
if __name__ == '__main__':
    env_metadata = MetadataSynthesis(
            num_qubits = 3, # As its name
            num_cnot = 14, # Number of CNOT gate you want
            depth = 30, # Ansatz depth you want
            num_circuit = 8, # Number of ansatz per generation
            num_generation = 10, # Number of generation/iteration for GA 
            prob_mutate = 0.05 # Mutation probability, usually as small as 0.05 (5%)
    )

    
    env = EEnvironment(
        metadata = env_metadata,
        fitness_func = fitness_synthesis,
        generator_func = generator.by_num_cnot,
        crossover_func = crossover.onepoint(
            divider.by_num_cnot(int(env_metadata.num_cnot/2)), 
            normalizer.by_num_cnot(env_metadata.num_cnot)),
        mutate_func = mutate.bitflip_mutate(constant.operations_only_cnot, env_metadata.prob_mutate),
        threshold_func= threshold.synthesis_threshold
    ).evol()