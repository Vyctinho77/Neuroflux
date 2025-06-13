import numpy as np
from neuroflux_adaptativa import NeurofluxAdaptativa
from regulador_harmonico import NeurofluxMemory

class Tokenizer:
    """Caracter-level tokenizer."""
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids)

def create_dataset(text, context=8):
    tok = Tokenizer(text)
    data = tok.encode(text)
    X, y = [], []
    for i in range(len(data) - context):
        X.append(data[i:i+context])
        y.append(data[i+context])
    X = np.array(X)
    y = np.array(y)
    return X, y, tok

def to_onehot(arr, vocab_size):
    eye = np.eye(vocab_size)
    return eye[arr].reshape(arr.shape[0], -1)

def to_onehot_targets(arr, vocab_size):
    eye = np.eye(vocab_size)
    return eye[arr]

class NeurofluxLLM:
    def __init__(self, text, context=8, hidden_size=128, chunk_size=32):
        X_idx, y_idx, self.tokenizer = create_dataset(text, context)
        self.vocab_size = self.tokenizer.vocab_size
        self.context = context

        X = to_onehot(X_idx, self.vocab_size)
        y = to_onehot_targets(y_idx, self.vocab_size)

        self.memory = NeurofluxMemory(X, y, chunk_size)
        self.model = NeurofluxAdaptativa(
            tamanho_entrada=context * self.vocab_size,
            tamanho_oculto=hidden_size,
            tamanho_saida=self.vocab_size
        )

    def train(self, epochs=500):
        for epoch in range(epochs):
            X_chunk, y_chunk = self.memory.next_chunk()
            loss = self.model.train_step(X_chunk, y_chunk)
            if epoch % 100 == 0:
                self.model.replace_neurons(0.25)
            if epoch % 50 == 0:
                print(f"Ep{epoch} loss:{loss:.4f}")

    def feed(self, text):
        """Add new text samples to memory for on-line learning."""
        X_idx, y_idx, _ = create_dataset(text, self.context)
        X = to_onehot(X_idx, self.vocab_size)
        y = to_onehot_targets(y_idx, self.vocab_size)
        self.memory.add_data(X, y)

    def generate(self, seed, length=100, temperature=1.0):
        context_idx = self.tokenizer.encode(seed)[-self.context:]
        output = list(context_idx)
        for _ in range(length):
            x = to_onehot(np.array([context_idx]), self.vocab_size)
            probs = self.model.forward(x)[0]
            probs = probs / probs.sum()
            if temperature != 1.0:
                probs = np.log(probs + 1e-8) / temperature
                probs = np.exp(probs) / np.exp(probs).sum()
            idx = np.random.choice(self.vocab_size, p=probs)
            output.append(idx)
            context_idx = context_idx[1:] + [idx]
        return seed + self.tokenizer.decode(output[self.context:])
