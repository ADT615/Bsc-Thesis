#Import required libraries
import matplotlib.pyplot as plt
import numpy as np

import sys, qiskit
from qiskit import QuantumCircuit
sys.path.insert(0, '../../..')
import numpy as np
import matplotlib.pyplot as plt
import qoop
from qoop.compilation.qsp import QuantumStatePreparation
from qoop.evolution import crossover, mutate, selection, threshold
from qoop.evolution.environment import EEnvironment, EEnvironmentMetadata
from qoop.core import state
from qoop.backend import tools, utilities
import model
import csv
from scipy.linalg import expm
from qiskit.quantum_info import Pauli, SparsePauliOp

#mod
mod = 'mod1'

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
    elif mod == 'mod8':
        u=-0.25
        h=[-0.25,-0.25]         
    return N, J, u, h, T   
    
def h_time(t):
    N, J, u, h, T = coefs(mod)
    return model.XYZ_model(N,J,u,h,T,t)

def trotter_circuit(nqubits, labels, coeffs, t, M):

    # Convert one Trotter decomposition ,e^{iZ_1Z_2*delta}*e^{iZ_2Z_3*delta}*...e^{iZ_nZ_1*delta} to a quantum gate
    circuit = QuantumCircuit(nqubits)
    for qubit in range(nqubits//2, nqubits):
        circuit.x(qubit)

    # Time increment range
    delta = 0.5
        
    for i in range(len(labels)):
        # 'IX', 'IZ', 'IY' case
        if labels[i][0] == 'I':
            if labels[i][1] == 'Z':
                circuit.rz(2*delta*coeffs[i],1)
            elif labels[i][1] == 'X':
                circuit.rx(2*delta*coeffs[i],1)
            elif labels[i][1] == 'Y':
                circuit.ry(2*delta*coeffs[i],1)
    
        # 'XI', 'ZI', 'YI' case
        elif labels[i][1] == 'I':
            if labels[i][0] == 'Z':
                circuit.rz(2*delta*coeffs[i],0)
            elif labels[i][0] == 'X':
                circuit.rx(2*delta*coeffs[i],0)
            elif labels[i][0] == 'Y':
                circuit.ry(2*delta*coeffs[i],0)
    
        # # 'XX', 'ZZ', 'YY' case
        elif labels[i] in ['XX', 'YY', 'ZZ']:
            for j in range(nqubits):
                if labels[i][1] == 'Z':
                    #circuit.cx((j+1)%(nqubits),j)
                    circuit.rzz(2*delta*coeffs[i],(j+1)%nqubits, j) ## RZ(a)=exp(i*a/2*Z)
                    #circuit.cx((j+1)%(nqubits),j)
                elif labels[i][1] == 'X':
                    #circuit.cx((j+1)%(nqubits),j)
                    circuit.rxx(2*delta*coeffs[i],(j+1)%nqubits, j) ## RZ(a)=exp(i*a/2*Z)
                    #circuit.cx((j+1)%(nqubits),j)
                elif labels[i][1] == 'Y':
                    #circuit.cx((j+1)%(nqubits),j)
                    circuit.ryy(2*delta*coeffs[i],(j+1)%nqubits, j) ## RZ(a)=exp(i*a/2*Z)
                    #circuit.cx((j+1)%(nqubits),j)
    return circuit

def load_data_csv(fname):
    # Define the file name
    csv_file = f'{mod}_theta.csv'
    loaded_qc = utilities.load_circuit(fname)
    thetass = []
    
    # Read data from the CSV file
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Assuming theta values are stored as floats in the CSV file
            theta_values = [float(value) for value in row]
            thetass.append(theta_values)
    return loaded_qc, thetass

def plot_metrics(qcs,thetass):
    mag_ga = []
    mag_theo = [] 
    mag_trotter = []
    
    #get coefs
    N, J, u, h, T = coefs(mod)
    #ZI = SparsePauliOp(['ZI'], coeffs=[1+0.j])
    #IZ = SparsePauliOp(['IZ'], coeffs=[1+0.j])
    #op_Z = 0.5*(ZI - IZ)
    op_Z = model.pauli_oper(N,oper = 'Z') #use this back
    
    
    #for theory
    times_theo = np.linspace(0,10,100)  
    for i, time in enumerate(times_theo): 
        qc_theo = model.time_dependent_qc(N,h_time,time)
        exp_theo = tools.get_expectation(qc_theo,op_Z,None)   
        mag_theo.append(exp_theo) 
    
    #for ga       
    times_ga = np.linspace(0,10,100)  
    for i, time in enumerate(times_ga):     
        qc_ga = qcs
        qc_p = qc_ga.assign_parameters(thetass[i])
        #sv = np.array(qi.Statevector(qc_p))
        exp_ga = tools.get_expectation(qc_p,op_Z,None)   
        mag_ga.append(exp_ga)

    #for trotter
    times_trotter = np.linspace(0,10,100)
    for i,time in enumerate(times_trotter):
        labels = model.time_dependent_integral(h_time,t=time).paulis.to_labels()
        coeffs = model.time_dependent_integral(h_time,t=time).coeffs
        coeffs = np.real(coeffs)
        #print(coeffs)
        qc = trotter_circuit(N,labels,coeffs, time, M=100)
        #print(qc)
        exp_trotter = tools.get_expectation(qc,op_Z,None)   
        mag_trotter.append(exp_trotter)

    print(qc.depth())
    

    # Save mag_trotter to a CSV file
    with open(f'trotter_{mod}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Expectation Values"])  # Optional header
        for value in mag_trotter:
            writer.writerow([value])
    
    # Calculate the mean square error
    error_ga = np.real(np.array(mag_theo) - np.array(mag_ga))**2
    error_trotter = np.real(np.array(mag_theo) - np.array(mag_trotter))**2 
    plt.scatter(times_trotter, error_trotter, label="trotter", marker='v')
    plt.scatter(times_ga,error_ga,label='GA', marker='^')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('square error')
    plt.savefig(f'error_{mod}')
    plt.savefig(f'error_{mod}.eps', format='eps')
    plt.close()
    #mse = np.sum(error)/len(error)
    #print('mse= ', mse)

    plt.plot(times_theo,mag_theo,label='mag theo')  
    plt.scatter(times_trotter, mag_trotter, label="trotter", marker='v')
    plt.scatter(times_ga,mag_ga,label='mag ga',marker='^')

    plt.xlabel('time')
    plt.ylabel('magnetization')

    #AVQDS
    # Specify the file path
    csv_file = f"magnetization_data_{mod}.csv"

    # Load data using NumPy
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)  # Skip header row
    
    # Extract columns into separate lists (if needed)
    times_avqds = data[:, 0]  # First column (Time)
    mag_avqds = data[:, 1]   # Second column (Magnetization)
    plt.plot(times_avqds,mag_avqds,label='mag AVQDS')  
    plt.legend()
    plt.savefig(f'mag_{mod}.eps', format='eps')
    plt.savefig(f'mag_{mod}')
    
#main   
if __name__ == '__main__':
    fname = f'n={mod},d=4,n_circuit=8,n_gen=16/best_circuit'
    #loaded_qc, thetass = load_data(fname)  #run it if not csv yet
    loaded_qc, thetass = load_data_csv(fname) #run it if you have csv    
    plot_metrics(loaded_qc,thetass)
