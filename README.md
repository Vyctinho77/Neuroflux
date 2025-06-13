<div align="center">
  <img src="https://github.com/user-attachments/assets/cd7a44d3-29f9-4399-878f-2b88478fe038" width="700"/>
</div>

---

## Overview

**Neuroflux** is an artificial neural network architecture designed to learn stably and adaptively, even in scenarios with weight oscillations and frequent changes in internal representations. The inspiration comes from electronic voltage regulator circuits, games with block loading like Minecraft, and self-stabilizing principles.

Unlike conventional networks that update all weights with a single rule, Neuroflux monitors its own instability and dynamically adjusts how it learns, maintaining healthy areas of the network while smoothing out turbulent zones.

### Motivation

Traditional neural networks can easily get lost in unstable regions of weight space—especially in logic tasks, state transformation, or rule manipulation. Neuroflux was created to mitigate this, offering:

- Weight regulation inspired by electrical harmony  
- Continuous evolution of neurons throughout the network  
- Memory that adapts with learning, preserving what is truly understood  
- Dynamic learning adjustments based on the network's internal instability  

### General Architecture

```
Input  →  Hidden Layer (with adaptive evolution)
             ↓
      Harmonic weight regulator
             ↓
   Learning rate adjustment by instability
             ↓
    Output with continuous activation function
```

<div align="center">
  <img src="https://github.com/user-attachments/assets/fa40220a-5098-47ee-ba4d-e6ede18547be" width="700"/>
</div>

### Key Equations

1. **Harmonic Weight Regulation**
```python
harmonico = np.tanh(deltas) + relu(deltas) * 0.5
```

2. **Oscillation Instability**
```
I(t) = (1/n) * Σ |Δw_i(t) - Δw_i(t-1)|
```

3. **Responsive Learning Rate**
```
η(t) = η_base * exp(-α * I(t))
```

### Neural Evolution Module

Neuroflux does not require an increase in the number of neurons over time. Instead, it:
- Evaluates the average performance of neurons  
- Replaces the least active with new candidates (random weights)  
- Keeps the architecture lean but smart  

<div align="center">
  <img src="https://github.com/user-attachments/assets/4cc5c789-ea5b-44fb-bcb2-8552d93c9e9e" width="700"/>
</div>

### Short-Term Memory with Chunk Loading

Inspired by the "chunk loading" concept from games, Neuroflux's memory system:
- Loads pieces of experience (chunks) each epoch  
- Updates its local perception of learning  
- Ensures the network does not depend on a large dataset fully loaded into memory  

<div align="center">
  <img src="https://github.com/user-attachments/assets/e3327124-c7d0-46ba-a94c-8844fca788c0" width="700"/>
</div>

### Applications and Tested Cases

- **Logic learning** (XOR, AND)  
- **Action planning** (Tower of Hanoi)  
- **3D transformations** (Pocket Cube / mini Rubik’s Cube)  

Each case showed that the network can maintain stability, growing accuracy, and flexibility when dealing with oscillating or unpredictable data.

### Installation

```bash
git clone https://github.com/Vyctinho77/Neuroflux.git
cd Neuroflux
python examples/xor_test.py
# Segmentation training (U-Net)
python train.py
```

### Directory Structure

```
Neurofluxo/
├── neuroflux_core.py
├── neuroflux_adaptativa.py
├── regulador_harmonico.py
├── regulador_instabilidade.py
├── memoria_chunk.py
├── utils.py
├── modelo.py
├── train.py
├── exemplos/
│   ├── xor_test.py
│   ├── hanoi_test.py
│   └── pocket_cube_test.py
└── README.md
```

> If you have any questions, suggestions, or are interested in collaborating, feel free to open an issue or send a message!

### Glioblastoma Segmentation with U-Net

This repository includes a multimodal U-Net with Grad-CAM for GBM segmentation.
To train on Google Colab with GPU, simply run:

```python
!git clone https://github.com/Vyctinho77/Neuroflux.git
%cd Neuroflux
!python train.py
```
---

### Neuroflux LLM

O projeto agora inclui um modelo de linguagem leve, construído inteiramente com o núcleo Neuroflux. Ele utiliza tokenização em nível de caractere e memória baseada em blocos, capaz de ingerir novos fluxos de texto. Um modo opcional de *tecido neural* permite que múltiplos blocos adaptativos evoluam em paralelo, apoiando a criatividade contínua.

``` python
de neuroflux _llm importar NeurofluxLLM

texto = "a mente aprende. o neurofluxo se adapta em tempo real." * 3
llm = NeurofluxLLM(texto, contexto=4, token_level ='palavra')
llm.train(épocas=200)
llm.feed("IA adaptativa em tempo real")
print(llm.generate("o", comprimento=20))
```

Para permitir a evolução dos tecidos neurais:

``` python
llm = NeurofluxLLM(texto, contexto=3, use _tecido=Verdadeiro, token_ nível='palavra')
llm.train(épocas=100)
print(llm.generate("cre", comprimento=30))
```

### Demonstração de raciocínio simbólico

O tecido neural pode operar puramente em caracteres e gerar raciocínio
sem um conjunto de dados externo. Um bloco especial *logo* explica suas ações.

``` python
alfabeto = "abcdefghijklmnopqrstuvwxyz"
llm = NeurofluxLLM(alfabeto, contexto=2, use _tecido=Verdadeiro)
llm.train(épocas=50)
print(llm.generate("ab?", comprimento=1))
imprimir(llm.logos())
```

### Demonstração de Ensino Semântico

O Neuroflux pode aprender conceitos por meio de pares de perguntas e respostas explícitas. Um pequeno
O conjunto de dados `semantics.jsonl` demonstra como alimentar significados durante a coleta
Explicações de `logos()`.

```python
importar json
de neuroflux_ llm importar NeurofluxLLM

com open('data/semantics.jsonl') como f:
    pares = [json.loads(l) para l em f]

text = " ".join(p['entrada'] + ' ' + p['explicacao'] para p em pares)
llm = NeurofluxLLM(texto, contexto=3, use _tecido=Verdadeiro, token_ nível='palavra')
llm.teach _semantics(pares, épocas=20)
print(llm.generate('O que', length=10))
imprimir(llm.logos())
```

### Ciclo de Ensino Supervisionado

`supervision.py` integra a API OpenAI para correção online. O LLM
gera uma resposta, o GPT‑3.5 Turbo a refina e o par corrigido é alimentado
de volta usando `teach_ semantics()`.

```python
de neuroflux_llm importar NeurofluxLLM
da supervisão importar passo_supervisionado

question = "O que é mover?"
llm = NeurofluxLLM(pergunta, contexto=3, nível_de_token='palavra')
raw = llm.generate(pergunta, comprimento=6)
corrigido = passo_supervisionado(pergunta, bruto)
llm.teach_semantics([{"entrada": pergunta, "explicação": corrigido}])
```

### Memória de streaming

`flex_context=True` habilita um buffer de atenção expansível. O modelo mantém um
histórico de todos os tokens que viu, permitindo que o contexto se estenda além do
tamanho inicial. O novo texto pode ser inserido de forma incremental e o histórico completo influencia
geração.

```python
de neuroflux_llm importar NeurofluxLLM

texto = "a curiosidade impulsiona a evolução"
llm = NeurofluxLLM(texto, contexto=3, use_tissue=True,
                   token_level='palavra', flex_context=True)
llm.train(épocas=30)
llm.feed("a evolução alimenta a curiosidade")
print(llm.generate("curiosidade", comprimento=6))
```

### Licença

Licença MIT

### Autor

Desenvolvido por **Vyctor** , 2025.

