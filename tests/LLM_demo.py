# LLM demo with Neuroflux
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from neuroflux_llm import NeurofluxLLM

# Texto base para treinar o modelo
text = "hello neuroflux world " * 5

# Inicializa o modelo LLM adaptativo
llm = NeurofluxLLM(text, context=4, hidden_size=64, chunk_size=8)

# Treina o modelo com os dados iniciais
llm.train(epochs=200)

# Adiciona novas frases (aprendizado incremental)
llm.feed("adaptive ai in realtime ")
llm.train(epochs=100)

# Gera texto a partir de um seed
sample = llm.generate("hell", length=40, temperature=0.8)
print("\nGenerated text:\n", sample)
