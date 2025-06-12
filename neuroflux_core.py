import numpy as np
from regulador_harmonico import regulador_harmonico
from utils import binary_cross_entropy

class NeurofluxCore:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        self.hist_W1 = np.zeros_like(self.W1)
        self.hist_W2 = np.zeros_like(self.W2)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return 1 / (1 + np.exp(-self.z2))

    def replace_neurons(self, rate=0.25):
        n_replace = int(self.W1.shape[1] * rate)
        idx = np.argsort(np.mean(np.abs(self.W1), axis=0))[:n_replace]
        self.W1[:, idx] = np.random.randn(self.W1.shape[0], n_replace) * 0.1
