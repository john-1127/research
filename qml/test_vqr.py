import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.optimizers import COBYLA, L_BFGS_B
from qiskit_machine_learning.utils import algorithm_globals

from qiskit_machine_learning.algorithms import VQR
from qiskit_machine_learning.circuit.library import QNNCircuit

algorithm_globals.random_seed = 42

from qiskit.primitives import StatevectorEstimator as Estimator
# from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
# from qiskit_aer.primitives import EstimatorV2

# service = QiskitRuntimeService(channel="ibm_quantum")
# backend = service.backend("ibm_brisbane")
# print(backend.configuration().max_shots)
# estimator = EstimatorV2(mode=backend, shots=1)
estimator = Estimator()

def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

# data
num_samples = 20
eps = 0.2
lb, ub = -np.pi, np.pi
X_ = np.linspace(lb, ub, num=50).reshape(50, 1)
f = lambda x: np.sin(x)

X = (ub - lb) * algorithm_globals.random.random([num_samples, 1]) + lb
y = f(X[:, 0]) + eps * (2 * algorithm_globals.random.random(num_samples) - 1)
param_x = Parameter("x")
feature_map = QuantumCircuit(1, name="fm")
feature_map.ry(param_x, 0)

# construct simple ansatz
param_y = Parameter("y")
ansatz = QuantumCircuit(1, name="vf")
ansatz.ry(param_y, 0)

# transpile for hardware
# from qiskit import transpile
# feature_map = transpile(feature_map, backend=backend)
# ansatz = transpile(ansatz, backend=backend)


# construct QNN
vqr = VQR(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=L_BFGS_B(maxiter=5),
    estimator=estimator,
)

objective_func_vals = []


vqr.fit(X, y)

print(vqr.score(X, y))
