# tests/tissue_demo.py
# Demonstração de tecido neural dinâmico com Neuroflux LLM

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from neuroflux_llm import NeurofluxLLM

# Corpus inicial com frases curtas
text = "creative tissues evolve. open evolution drives change." * 2

# Cria LLM com tecido neural e tokenização por palavra
llm = NeurofluxLLM(
    text,
    context=3,
    hidden_size=32,
    chunk_size=4,
    use_tissue=True,
    token_level='word'
)

# Treinamento inicial
llm.train(epochs=50)

# Alimenta o modelo com novos dados e retreina progressivamente
for _ in range(3):
    llm.feed("open evolution drives adaptation")
    llm.train(epochs=30)

# Gera texto a partir de uma semente curta
print("Generated:\n", llm.generate("cre", length=30))
