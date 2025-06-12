import numpy as np

class ReguladorInstabilidadeAdaptativa:
    def __init__(self, alpha=1.5, eta_base=0.1):
        self.alpha = alpha
        self.eta_base = eta_base
        self.w_anterior = None
        self.w_penultimo = None

    def atualizar(self, w_atual):
        if self.w_anterior is None:
            self.w_anterior = w_atual.copy()
            self.w_penultimo = w_atual.copy()
            return self.eta_base

        delta_atual = w_atual - self.w_anterior
        delta_anterior = self.w_anterior - self.w_penultimo
        oscilacao = np.abs(delta_atual - delta_anterior)
        instabilidade = np.mean(oscilacao)

        eta_t = self.eta_base * np.exp(-self.alpha * instabilidade)

        self.w_penultimo = self.w_anterior.copy()
        self.w_anterior = w_atual.copy()

        return eta_t
