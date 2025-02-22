import torch as nn
import pennylane as qml

n_wires = 4
n_layers = 1

dev = qml.device('default.qubit', wires=n_wires)

params_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires)
params = nn.rand(params_shape)

@qml.qnode(dev, interface='torch', diff_method="backprop")
def circuit_cuda(params):
    qml.StronglyEntanglingLayers(params, wires=range(n_wires))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]


import timeit
# print(timeit.timeit("circuit_cuda(params)", globals=globals(), number=5))
params = params.to(device=nn.device('cuda'))
print(timeit.timeit("circuit_cuda(params)", globals=globals(), number=550))


