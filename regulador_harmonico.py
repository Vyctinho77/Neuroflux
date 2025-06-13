import numpy as np

def regulador_harmonico(dW, hist):
    fator_linear = dW
    fator_nonlinear = np.tanh(dW - hist)
    return 0.5 * fator_linear + 0.5 * fator_nonlinear

class NeurofluxMemory:
    def __init__(self, X, y, chunk_size=4, shuffle=True):
        self.X = np.array(X)
        self.y = np.array(y)
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.index = 0
        if self.shuffle:
            self._shuffle()

    def _shuffle(self):
        perm = np.random.permutation(len(self.X))
        self.X = self.X[perm]
        self.y = self.y[perm]

    def add_data(self, X_new, y_new):
        self.X = np.vstack([self.X, np.array(X_new)])
        self.y = np.vstack([self.y, np.array(y_new)])
        if self.shuffle:
            self._shuffle()

    def next_chunk(self):
        # Checa se restam poucos dados para um chunk completo
        if self.index >= len(self.X):
            self.index = 0
            if self.shuffle:
                self._shuffle()
        end = min(self.index + self.chunk_size, len(self.X))
        X_chunk = self.X[self.index:end]
        y_chunk = self.y[self.index:end]
        self.index = end
        return X_chunk, y_chunk

