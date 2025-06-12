# Pocket Cube Teste com Neuroflux Adaptativa
import numpy as np
from neuroflux_adaptativa import NeurofluxAdaptativa
from memoria_chunk import NeurofluxMemory

# Exemplo simplificado de estados de Pocket Cube
X = np.array([
    [0.2]*24,
    [0.3]*24,
    [0.4]*24,
    [0.5]*24
])
y_state = np.array([
    [0.3]*24,
    [0.4]*24,
    [0.5]*24,
    [0.2]*24
])
y_mov = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
])
y_resolved = np.array([
    [0.1]*24,
    [0.1]*24,
    [0.1]*24,
    [0.1]*24
])

# Criar memórias
mem_fw = NeurofluxMemory(X, y_state, chunk_size=2)
mem_mv = NeurofluxMemory(X, y_mov, chunk_size=2)
mem_rv = NeurofluxMemory(X, y_resolved, chunk_size=2)

# Núcleos
core_fw = NeurofluxAdaptativa(24, 48, 24)
core_mv = NeurofluxAdaptativa(24, 32, 3)
core_rv = NeurofluxAdaptativa(24, 48, 24)

# Treinar
for epoch in range(500):
    core_fw.train_step(*mem_fw.next_chunk())
    core_mv.train_step(*mem_mv.next_chunk())
    core_rv.train_step(*mem_rv.next_chunk())
    if epoch % 150 == 0:
        core_fw.replace_neurons(0.25)
        core_mv.replace_neurons(0.25)
        core_rv.replace_neurons(0.25)

# Testar
pred_fw = core_fw.forward(X) * 5
pred_mv = core_mv.forward(X)
pred_rv = core_rv.forward(X) * 5

print("\nPocket Cube Teste:")
for i in range(len(X)):
    print(f"Estado: {np.round(X[i]*5).astype(int).tolist()} => Próximo: {np.round(pred_fw[i]).astype(int).tolist()} | Movimento: {np.argmax(pred_mv[i])} | Resolvido: {np.round(pred_rv[i]).astype(int).tolist()}")
