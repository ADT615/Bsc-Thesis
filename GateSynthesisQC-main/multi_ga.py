import qiskit
from qoop.core import state, metric, ansatz
from qoop.compilation.qsp import QuantumStatePreparation
from qoop.evolution.environment_compilation import MetadataCompilation
from qoop.evolution.environment import EEnvironment
from qoop.evolution import crossover, divider, normalizer

def fitnessW(qc: qiskit.QuantumCircuit):
    qsp = QuantumStatePreparation(
        u = qc,
        target_state = state.w(num_qubits = 3).inverse()
    ).fit(num_steps = 10)
    return 1 - qsp.compiler.metrics['loss_fubini_study'][-1] # Fitness value

env_metadata = MetadataCompilation(
        num_qubits = 3, # As its name
        depth = 4, # Ansatz depth you want
        num_circuit = 4, # Number of ansatz per generation
        num_generation = 3, # Number of generation/iteration for GA 
        prob_mutate = 0.01 # Mutation probability, usually as small as 0.01 (1%)
)

env = EEnvironment(
    metadata = env_metadata,
    fitness_func = fitnessW,
    crossover_func= crossover.onepoint(divider.by_depth(2), normalizer.by_depth(4))
).evol()