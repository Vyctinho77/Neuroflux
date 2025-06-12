from regulador_instabilidade import ReguladorInstabilidadeAdaptativa

class NeurofluxAdaptativa(NeurofluxCore):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(input_size, hidden_size, output_size)
        self.regulador_W1 = ReguladorInstabilidadeAdaptativa()
        self.regulador_W2 = ReguladorInstabilidadeAdaptativa()

    def train_step(self, X, y):
        out = self.forward(X)
        loss = binary_cross_entropy(out, y)

        dL_dz2 = out - y
        dW2_raw = self.a1.T @ dL_dz2 / len(X)
        db2 = np.mean(dL_dz2, axis=0, keepdims=True)

        dL_da1 = dL_dz2 @ self.W2.T
        dL_dz1 = dL_da1 * (self.z1 > 0)
        dW1_raw = X.T @ dL_dz1 / len(X)
        db1 = np.mean(dL_dz1, axis=0, keepdims=True)

        dW2 = regulador_harmonico(dW2_raw, self.hist_W2)
        dW1 = regulador_harmonico(dW1_raw, self.hist_W1)

        eta_W1 = self.regulador_W1.atualizar(self.W1.flatten())
        eta_W2 = self.regulador_W2.atualizar(self.W2.flatten())

        self.W2 -= eta_W2 * dW2
        self.b2 -= 0.01 * db2
        self.W1 -= eta_W1 * dW1
        self.b1 -= 0.01 * db1

        self.hist_W1 = 0.9 * self.hist_W1 + 0.1 * dW1
        self.hist_W2 = 0.9 * self.hist_W2 + 0.1 * dW2
        return loss
