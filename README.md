<div align="center">
  <img src="https://github.com/user-attachments/assets/cd7a44d3-29f9-4399-878f-2b88478fe038" width="700"/>
</div>

---

# Neuroflux

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

texto = "olá mundo neuroflux" * 5
llm = NeurofluxLLM(texto, contexto=4)
llm.train(épocas=200)
llm.feed("IA adaptativa em tempo real")
print(llm.generate("inferno", comprimento=40))
```

Para permitir a evolução dos tecidos neurais:

```python
llm = NeurofluxLLM(texto, contexto=3, use_tecido =Verdadeiro)
llm.train(épocas=100)
print(llm.generate("cre", comprimento=30))
