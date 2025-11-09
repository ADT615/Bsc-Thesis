import sys, qiskit
sys.path.insert(0, '../../..')
import matplotlib.pyplot as plt
import numpy as np
import qoop
from qoop.compilation.qsp import QuantumStatePreparation
from qoop.core import ansatz, state, random_circuit
from qoop.backend import constant, utilities
from qoop.evolution import crossover, mutate, selection, threshold
from qoop.evolution.environment import EEnvironment, EEnvironmentMetadata
import pickle
import model
import os

# def system coefs
def coefs(mod):
    #return coefs
    N=2
    J=1
    T=10
    #test mod
    if mod == 'mod1':  
        u=0
        h=[0,0]
    elif mod == 'mod2':
        u=-1
        h=[0.25,0.25]
    elif mod == 'mod3':
        u=1
        h=[0.25,0.25]
    elif mod == 'mod4':
        u=0.25
        h=[0.25,0.25]  
    elif mod == 'mod5':
        u=-0.25
        h=[0.25,0.25]  
    elif mod == 'mod6':
        u=-0.25
        h=[0,0]    
    elif mod == 'mod7':
        u=0.0
        h=[0.25,0.25] 
    elif mod == 'mod8': #same as 5
        u=-0.25
        h=[-0.25,-0.25]         
    return N, J, u, h, T     

mod = 'mod5'     
def h_time(t):
    N, J, u, h, T = coefs(mod)
    return model.XYZ_model(N,J,u,h,T,t)
    
def compilation_fitness(qc: qiskit.QuantumCircuit):
    p0s = []
    N, J, u, h, T = coefs(mod) 
    times = np.linspace(0,10,100)
    for time in times:
        qsp = QuantumStatePreparation(
            u=qc,
            target_state=model.time_dependent_qc(N,h_time,time).inverse()
            ).fit(num_steps=300, metrics_func=['loss_basic'])
        p0s.append(1-qsp.compiler.metrics['loss_basic'][-1])
        time_folder = os.path.join(f"times_{mod}/times= {time}")
        os.makedirs(time_folder, exist_ok=True)
        qsp.save(f"times_{mod}/times= {time}")
        
    return np.mean(p0s)

def super_evol(_depth, _num_circuit, _num_generation):
    env_metadata = EEnvironmentMetadata(
        num_qubits = num_qubits,
        depth = _depth,
        num_circuit = _num_circuit,
        num_generation = _num_generation,
        prob_mutate=3/(_depth * _num_circuit)
    )
    env = EEnvironment(
        metadata = env_metadata,
        fitness_func= compilation_fitness,
        selection_func=selection.elitist_selection,
        crossover_func=crossover.onepoint_crossover,
        mutate_func=mutate.layerflip_mutate,
        threshold_func=threshold.compilation_threshold
    )
    env.set_filename(f'n={mod},d={_depth},n_circuit={_num_circuit},n_gen={_num_generation}')
    env.evol()
    print('done')
    
# main
if __name__ == '__main__':
    num_qubits = 2
    super_evol(4,8,16)









