# regulador_harmonico.py
import numpy as np

def regulador_harmonico(dW, hist):
    """Aplica regulação harmônica entre gradiente e histórico."""
    fator_linear = dW
    fator_nonlinear = np.tanh(dW - hist)
    return 0.5 * fator_linear + 0.5 * fator_nonlinear


# memoria_chunk.py
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
        """Embaralha os dados."""
        perm = np.random.permutation(len(self.X))
        self.X = self.X[perm]
        self.y = self.y[perm]

    def add_data(self, X_new, y_new):
        """Adiciona novos dados à memória e embaralha, se necessário."""
        self.X = np.vstack([self.X, np.array(X_new)])
        self.y = np.vstack([self.y, np.array(y_new)])
        if self.shuffle:
            self._shuffle()

    def next_chunk(self):
        """Retorna o próximo chunk de dados."""
        if self.index + self.chunk_size > len(self.X):
            self.index = 0
            if self.shuffle:
                self._shuffle()
        X_chunk = self.X[self.index:self.index + self.chunk_size]
        y_chunk = self.y[self.index:self.index + self.chunk_size]
        self.index += self.chunk_size
        return X_chunk, y_chunk


