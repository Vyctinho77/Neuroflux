# Testes da Neuroflux

Este diretório contém os scripts de teste que demonstram as capacidades adaptativas e cognitivas da arquitetura **Neuroflux**. Cada script foca em um tipo diferente de tarefa, todas com o objetivo de validar a regulação harmônica e a evolução neural inspirada em chunk loading e instabilidade adaptativa.

---

## Estrutura dos Testes

### `xor_test.py`
- **Objetivo:** Validar a capacidade da Neuroflux de aprender operações lógicas básicas (XOR).
- **Por que importa:** A maioria das redes simples falha ao aprender XOR sem camadas escondidas ou regulação apropriada.
- **Como funciona:**
  - Dados binários são apresentados.
  - A rede treina com backpropagation adaptativo.
  - Verifica-se se a saída se aproxima de [0, 1, 1, 0].

### `hanoi_test.py`
- **Objetivo:** Treinar a rede a prever próximos estados na Torre de Hanói (3 discos).
- **Inspiração:** Testes de sequência realizados por Apple e DeepMind em arquiteturas de memória neural.
- **Como funciona:**
  - Os estados dos discos são codificados numericamente.
  - A rede aprende a sequência correta.
  - Demonstra planejamento reverso e aprendizado de regras.

### `pocket_cube_test.py`
- **Objetivo:** Testar uma tarefa 3D (2x2x2 Rubik's Cube).
- **Como funciona:**
  - Usa representações de estados como vetores com 24 elementos.
  - A rede prevê o próximo estado, movimento aplicado e estado resolvido.
  - Mostra a eficiência da regulação de instabilidade adaptativa.

---

## Inspirações e Pesquisas

Estes testes foram inspirados em:
- **Apple (Machine Learning Research)**: testes com sequências simbólicas, ordenamento, memória reversível.
- **DeepMind**: redes neurais com memória (DNC, NTM) para aprendizado de regras e operações.
- **Minecraft Chunk Loading**: inspirou o sistema de evolução de neurônios em linha do tempo.
- **Circuitos eletrônicos e reguladores de tensão**: inspiraram a regulação harmônica de pesos e estabilidade.

---

## Como Executar
```bash
python exemplos/xor_test.py
python exemplos/hanoi_test.py
python exemplos/pocket_cube_test.py
```

---

## Resultados Esperados
- Os testes imprimem previsões da rede para comparação humana.
- Pequenas diferenças podem existir, pois a rede aprende via regulação suave.
- A instabilidade é contida automaticamente e a precisão aumenta com o tempo.

---

Desenvolvido por Vyctor, 2025.
