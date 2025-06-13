# Neuroflux

> Uma arquitetura neural contínua com regulação de pesos responsiva, memória viva e atualização evolutiva de neurônios.

---

## Visão Geral

**Neuroflux** é uma arquitetura de rede neural artificial projetada para aprender de forma estável e adaptativa mesmo em cenários onde há oscilações nos pesos e mudanças frequentes nas representações internas. A inspiração vem de circuitos eletrônicos de reguladores de tensão, jogos com carregamento de blocos como Minecraft e princípios de autoestabilização.

Ao contrário das redes convencionais que atualizam todos os pesos com uma única regra, Neuroflux observa seu próprio estado de instabilidade e ajusta dinamicamente a forma como aprende, mantendo áreas saudáveis da rede enquanto suaviza zonas turbulentas.

---

## Motivação

Redes neurais tradicionais podem facilmente se perder em regiões instáveis do espaço de pesos — especialmente em tarefas de lógica, transformação de estado ou manipulação de regras. Neuroflux surge como uma proposta para mitigar isso com:

- Regulação de pesos inspirada em harmonia elétrica;
- Evolução contínua dos neurônios ao longo da rede;
- Memória que se adapta com o aprendizado, preservando o que realmente foi compreendido;
- Ajustes dinâmicos de aprendizado baseados na instabilidade interna da rede.

---

## Arquitetura Geral

```
Entrada  →  Camada Oculta (com evolução adaptativa)
            ↓
          Regulador de pesos harmônico
            ↓
      Ajuste da taxa de aprendizado por instabilidade
            ↓
        Saída com função de ativação contínua
```

<div align="center">
  <img src="https://github.com/user-attachments/assets/fa40220a-5098-47ee-ba4d-e6ede18547be" width="700"/>
</div>

---

## Equações-Chave

### 1. **Regulação Harmônica de Pesos**
Combinação de dois domínios:
- Linear (direção direta de erro)
- Não-linear (curvas suavizantes)

```python
harmonico = np.tanh(deltas) + relu(deltas) * 0.5
```

---

### 2. **Instabilidade por Oscilação**
Mede a diferença de variação entre épocas:

```
I(t) = (1/n) * Σ |Δw_i(t) - Δw_i(t-1)|
```

### 3. **Taxa de Aprendizado Responsiva**
Ajuste contínuo com base na instabilidade:

```
η(t) = η_base * exp(-α * I(t))
```

---

## Módulo de Evolução Neural

A Neuroflux não precisa de aumento no número de neurônios ao longo do tempo. Em vez disso, ela:
- Avalia o desempenho médio dos neurônios;
- Substitui os menos ativos por novos candidatos (com pesos randômicos);
- Mantém a arquitetura enxuta, mas inteligente.

<div align="center">
  <img src="https://github.com/user-attachments/assets/4cc5c789-ea5b-44fb-bcb2-8552d93c9e9e" width="700"/>
</div>

---

## Memória de Curto Prazo com Chunk Loading

Inspirado no conceito de "chunk loading" em jogos, o sistema de memória da Neuroflux:
- Carrega pedaços da experiência (chunks) a cada época;
- Atualiza sua percepção local de aprendizado;
- Garante que a rede não dependa de um grande conjunto de dados carregado inteiro na memória.

<div align="center">
  <img src="https://github.com/user-attachments/assets/e3327124-c7d0-46ba-a94c-8844fca788c0" width="700"/>
</div>

---

## Aplicações e Casos Testados

- **Aprendizado de lógica** (XOR, AND)
- **Planejamento de ações** (Torre de Hanói)
- **Transformações 3D** (Pocket Cube / mini cubo mágico)

Cada caso mostrou que a rede consegue manter estabilidade, precisão crescente e flexibilidade diante de dados oscilantes ou imprevisíveis.

---

## Instalação

```bash
git clone https://github.com/seu-usuario/Neuroflux.git
cd Neuroflux
python exemplos/xor_test.py
```

---

## Diretório

```
Neuroflux/
├── neuroflux_core.py
├── neuroflux_adaptativa.py
├── regulador_harmonico.py
├── regulador_instabilidade.py
├── memoria_chunk.py
├── utils.py
├── exemplos/
│   ├── xor_test.py
│   ├── hanoi_test.py
│   └── pocket_cube_test.py
└── README.md
```

---

> Em caso de dúvidas, sugestões ou interesse em colaboração, fique à vontade para abrir uma issue ou enviar uma mensagem!

---

## Segmentação de Glioblastoma com U-Net

Este repositório inclui uma U-Net multimodal com Grad-CAM para segmentação de GBM.
Para treinar no Google Colab com GPU basta executar:

```python
!git clone https://github.com/seu-usuario/Neuroflux.git
%cd Neurofluxo
! trem python.py
```

O script `train.py` carrega as imagens e máscaras, treina a rede e gera métricas
como Loss, Dice, Precisão, Sensibilidade e Especificidade, além de salvar o overlay
de Grad-CAM em `gradcam_example.png`.

---

## Licença
MIT License

---

## Autor
Desenvolvido por **Vyctor**, 2025.
