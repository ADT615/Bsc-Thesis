# Quantum Simulation - Modular Approach

Đây là project quantum simulation được tổ chức theo cách modular, chia code thành các file riêng biệt theo chức năng.

## Cấu trúc Project

```
/
├── config.py              # Configuration và constants
├── quantum_setup.py       # Setup molecular problem và quantum operators
├── ansatz.py             # Quantum circuit ansatz definitions
├── time_evolution.py     # Time evolution calculations
├── optimization.py       # VQE và optimization functions
├── visualization.py      # Plotting và visualization tools
├── utils.py              # Utility functions
├── main_workflow.py      # Complete workflow orchestration
├── modular_demo.ipynb    # Demo notebook sử dụng modular approach
├── requirements.txt      # Python dependencies
└── README.md            # Hướng dẫn này
```

## Installation

1. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

2. Hoặc cài đặt từng package:
```bash
pip install qiskit qiskit-nature qiskit-algorithms pennylane numpy scipy matplotlib pyscf
```

## Cách sử dụng

### 1. Approach 1: Import từng module cần thiết

```python
# Import các module
from quantum_setup import setup_molecular_problem, setup_qubit_operators
from vqe_optimization import run_vqe
from visualization import plot_energy_convergence

# Setup problem
problem, hamiltonian, dipole = setup_molecular_problem()
qubit_ops = setup_qubit_operators(hamiltonian, dipole)

# Run VQE
result = run_vqe(qubit_ops['H_0'], ansatz)

# Plot results
plot_energy_convergence(energies)
```

### 2. Approach 2: Chạy complete workflow

```python
from main_workflow import run_complete_workflow

# Chạy toàn bộ workflow
results = run_complete_workflow()
```

### 3. Approach 3: Sử dụng trong Jupyter Notebook

Xem file `modular_demo.ipynb` để có example đầy đủ.

## Modules Chi tiết

### config.py
Chứa tất cả các constants và configuration parameters:
- Physical parameters (E0, GAMMA)
- Molecular geometry 
- Optimization settings
- Time evolution parameters

### quantum_setup.py
Setup molecular problem và quantum operators:
- `setup_molecular_problem()`: Tạo molecular problem với PySCF
- `setup_qubit_operators()`: Map sang qubit operators
- `setup_ansatz()`: Tạo UCC ansatz
- `get_exact_ground_state()`: Tính exact ground state

### ansatz.py
Quantum circuit ansatz cho PennyLane:
- `pennylane_ansatz_from_qiskit_pauli_evo()`: Convert Qiskit ansatz sang PennyLane
- `create_pauli_labels_from_operators()`: Tạo Pauli labels từ operators

### time_evolution.py
Time evolution calculations:
- `E_field()`: Electric field function
- `H_t_matrix()`: Time-dependent Hamiltonian
- `compute_target_unitaries()`: Compute target unitaries với ODE solver
- `unitary_rhs()`: RHS của differential equation

### optimization.py
VQE và optimization:
- `run_vqe()`: Chạy VQE optimization
- `get_vqe_ground_state()`: Get ground state từ VQE result
- `unitary_fidelity()`: Tính fidelity giữa 2 unitaries
- `OptimizationTracker`: Class để track optimization progress

### visualization.py
Plotting và visualization:
- `plot_optimization_convergence()`: Plot convergence của optimization
- `plot_energy_convergence()`: Plot energy convergence
- `plot_fidelity_vs_time()`: Plot fidelity theo time
- `plot_electric_field()`: Plot electric field profile
- `plot_absorption_spectrum()`: Plot absorption spectrum

### utils.py
Utility functions:
- `save_results()`, `load_results()`: Save/load data
- `convert_numpy_to_json()`: Convert numpy arrays cho JSON
- `calculate_statistics()`: Tính statistics cơ bản
- `normalize_array()`: Normalize arrays
- `print_system_info()`: Print system information

### main_workflow.py
Complete workflow orchestration:
- `run_complete_workflow()`: Chạy toàn bộ workflow từ đầu đến cuối
- `save_workflow_results()`: Save results với timestamps

## Ưu điểm của Modular Approach

### 1. **Organization & Readability**
- Code được tổ chức rõ ràng theo chức năng
- Dễ navigate và tìm kiếm functions
- Reduced cognitive load khi đọc code

### 2. **Reusability**
- Functions có thể tái sử dụng trong projects khác
- Import chỉ những gì cần thiết
- Tránh code duplication

### 3. **Maintainability**
- Dễ debug vì có thể test từng module riêng
- Modify một module không ảnh hưởng đến modules khác
- Version control tốt hơn (conflicts ít hơn)

### 4. **Collaboration**
- Nhiều người có thể làm việc trên các modules khác nhau
- Clear separation of concerns
- Easier code review

### 5. **Testing**
- Có thể unit test từng function riêng biệt
- Mock dependencies dễ dàng
- Regression testing hiệu quả hơn

### 6. **Performance**
- Import chỉ những modules cần thiết
- Lazy loading possible
- Memory efficient hơn

## Examples

### Example 1: Setup và chạy VQE
```python
from quantum_setup import setup_molecular_problem, setup_qubit_operators, setup_ansatz
from vqe_optimization import run_vqe

# Setup
problem, hamiltonian, dipole = setup_molecular_problem()
qubit_ops = setup_qubit_operators(hamiltonian, dipole)
ansatz = setup_ansatz(problem, qubit_ops['mapper'])

# Run VQE
result = run_vqe(qubit_ops['H_0'], ansatz)
print(f"Ground state energy: {result.optimal_value}")
```

### Example 2: Time evolution
```python
from time_evolution import compute_target_unitaries
from visualization import plot_electric_field

# Compute evolution
unitaries, times = compute_target_unitaries(H_static, dipole_matrix)

# Visualize electric field
plot_electric_field(times)
```

### Example 3: Custom configuration
```python
import config

# Modify config
config.E0 = 0.02
config.GAMMA = 0.3
config.OPTIMIZER_MAXITER = 300

# Then run workflow
from main_workflow import run_complete_workflow
results = run_complete_workflow()
```

## So sánh với Notebook approach

### Notebook Approach (Trước)
- ✅ Interactive development
- ✅ Easy visualization
- ❌ Code duplication
- ❌ Hard to maintain
- ❌ Difficult collaboration
- ❌ No reusability

### Modular Approach (Sau)
- ✅ Clean organization
- ✅ Reusable code
- ✅ Easy testing
- ✅ Better collaboration
- ✅ Maintainable
- ✅ Still works with notebooks!

## Best Practices

1. **Import conventions**:
```python
# Import specific functions
from quantum_setup import setup_molecular_problem

# Import module
import config

# Import with alias
from visualization import plot_energy_convergence as plot_energy
```

2. **Configuration management**:
- Modify `config.py` cho global settings
- Pass parameters explicitly cho specific functions
- Use environment variables cho production settings

3. **Error handling**:
```python
try:
    results = run_complete_workflow()
except Exception as e:
    print(f"Workflow failed: {e}")
    # Handle error appropriately
```

4. **Data management**:
```python
from utils import save_results, load_results

# Save intermediate results
save_results(intermediate_data, 'checkpoint.json')

# Load for continuation
data = load_results('checkpoint.json')
```

## Troubleshooting

### Common Issues

1. **Import errors**: Đảm bảo working directory đúng
2. **Missing dependencies**: Chạy `pip install -r requirements.txt`
3. **Memory issues**: Sử dụng `utils.memory_usage()` để monitor
4. **Performance**: Profile từng module riêng để identify bottlenecks

### Getting Help

1. Check docstrings của functions
2. Xem examples trong `modular_demo.ipynb`
3. Use `print_system_info()` để debug environment issues
4. Enable verbose logging trong các functions

## Future Improvements

1. **Add logging**: Structured logging thay vì print statements
2. **Configuration validation**: Validate config parameters
3. **Parallel processing**: Parallelize time evolution calculations
4. **Caching**: Cache expensive computations
5. **Type hints**: Add type hints cho better IDE support
6. **Documentation**: Generate docs từ docstrings

---

Happy quantum computing! 🚀
