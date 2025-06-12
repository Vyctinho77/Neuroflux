# Torre de Hanói com Neuroflux Adaptativa
import numpy as np
from neuroflux_adaptativa import NeurofluxAdaptativa

# Exemplo simbólico: estados codificados como vetores
# Estado: [1, 2, 3] representa discos em torres
# Movimento: qual disco vai para qual torre
X_hanoi = np.array([
    [1, 1, 1],  # todos na torre A
    [1, 1, 2],
    [1, 3, 2],
    [3, 3, 2]   # final em torre C
])
y_hanoi = np.array([
    [1, 1, 2],
    [1, 3, 2],
    [3, 3, 2],
    [1, 1, 1]   # tentativa reversa
])

model = NeurofluxAdaptativa(input_size=3, hidden_size=16, output_size=3)

# Treinamento
for epoch in range(800):
    model.train_step(X_hanoi, y_hanoi)
    if epoch % 200 == 0:
        model.replace_neurons(0.25)

# Teste
pred = model.forward(X_hanoi)
print("\nPrevisão Hanói:")
for i, p in zip(X_hanoi, pred):
    print(f"Entrada: {i.tolist()} => Saída prevista: {np.round(p).astype(int).tolist()}")
