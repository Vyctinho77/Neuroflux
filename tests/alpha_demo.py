# tests/alpha_demo.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from neuroflux_llm import NeurofluxLLM

# Corpus com alfabeto puro (modo simbólico)
alphabet = "abcdefghijklmnopqrstuvwxyz"

# Inicializa a LLM com tokenização por caractere e tecido neural ativado
llm = NeurofluxLLM(
    alphabet,
    context=2,
    hidden_size=32,
    chunk_size=4,
    use_tissue=True
)

# Treinamento simbólico (apenas alfabeto)
llm.train(epochs=50)

# Teste de raciocínio: prever a próxima letra após 'a b ?'
result = llm.generate("ab?", length=1)

# Exibe o resultado e as explicações geradas
print("Result:", result)
print("Logos:", llm.logos())
