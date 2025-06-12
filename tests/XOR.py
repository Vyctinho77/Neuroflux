# XOR Teste com Neuroflux Adaptativa
import numpy as np
from neuroflux_adaptativa import NeurofluxAdaptativa

# Dados XOR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([[0], [1], [1], [0]])

# Ajustar para 2 entradas e 1 saída (binária)
model = NeurofluxAdaptativa(input_size=2, hidden_size=8, output_size=1)

# Treinar
for epoch in range(1000):
    model.train_step(X, y)
    if epoch % 200 == 0:
        model.replace_neurons(0.3)

# Testar
output = model.forward(X)
print("\nResultados XOR:")
for i, o in zip(X, output):
    print(f"Entrada: {i} => Saída: {o[0]:.4f}")
