import numpy as np

def regulador_harmonico(dW, hist):
    fator_linear = dW
    fator_nonlinear = np.tanh(dW - hist)
    return 0.5 * fator_linear + 0.5 * fator_nonlinear

# memoria_chunk.py
class NeurofluxMemory:
    def __init__(self, X, y, chunk_size=4):
        self.X = X
        self.y = y
        self.chunk_size = chunk_size
        self.index = 0

    def next_chunk(self):
        if self.index + self.chunk_size > len(self.X):
            self.index = 0
        X_chunk = self.X[self.index:self.index+self.chunk_size]
        y_chunk = self.y[self.index:self.index+self.chunk_size]
        self.index += self.chunk_size
        return X_chunk, y_chunk
