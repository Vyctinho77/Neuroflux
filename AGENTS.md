# AGENTS.md

## Neuroflux – Arquitetura de Agentes em Tecido Neural

Este documento descreve os principais **agentes funcionais** do sistema Neuroflux, organizados como **blocos especializados** que compõem o tecido neural adaptativo. Cada bloco (ou agente) possui um papel específico na aprendizagem contínua, raciocínio semântico e autorregulação do modelo.

---

## Estrutura Geral

O tecido neural do Neuroflux é composto por unidades chamadas **blocos**, que evoluem, interagem e se adaptam com base na instabilidade, desempenho e contexto. Os agentes abaixo representam **níveis especializados** desses blocos.

---

## Tipos de Agentes

### 1. **InputBlock**
Responsável por codificar a entrada recebida em tokens (caracteres ou palavras) e preparar o vetor de ativação inicial.

- Tokenização híbrida (char/word)
- Preenchimento de contexto
- Padronização da sequência para o tecido

---

### 2. **MemoryBlock**
Gerencia a **memória dinâmica em blocos** (chunked learning), permitindo que o modelo treine com fragmentos sucessivos.

- Curto prazo: contexto imediato da entrada
- Médio prazo: padrões recorrentes entre blocos
- Longo prazo: memória semântica consolidada
- Atualização contínua com prioridade por uso e instabilidade

---

### 3. **RegulationBlock**
Controla o ajuste dos pesos com base na **instabilidade local** da rede.

- Avalia flutuações Δw por época
- Aplica regulações harmônicas (mistura linear + não linear)
- Calcula taxa de aprendizado responsiva `η(t)`

---

### 4. **EvolutionBlock**
Executa a **substituição evolutiva de neurônios** com baixo desempenho ou impacto.

- Avaliação de desempenho por média de contribuição
- Seleção de candidatos aleatórios com pesos iniciais
- Mantém a arquitetura compacta e eficaz

---

### 5. **LogosBlock**
Bloco **explicativo** especializado em **produzir interpretações** ou justificativas das saídas do modelo.

- Ativa-se quando o contexto pede por sentido, causalidade ou intenção
- Acompanha raciocínios e decisões geradas
- Pode ser interrogado via `llm.logos()` para mostrar “por que” algo foi feito

---

### 6. **CreativeBlock**
Estimula a **exploração aleatória guiada** para gerar variedade e inovação.

- Utiliza entropia como motor de expansão
- Cria combinações semânticas improváveis
- Crucial em modos de geração livre ou raciocínio simbólico

---

### 7. **CuriosityDriver**
Mecanismo que dispara ciclos de aprendizado **em regiões instáveis ou pouco compreendidas**.

- Detecta padrões incompletos ou conflitantes
- Incentiva expansão local do tecido
- Gera blocos derivados de forma emergente

---

## Fluxo de Ativação

```text
[InputBlock] → [MemoryBlock] → [RegulationBlock]
       ↓               ↓               ↓
 [EvolutionBlock] → [LogosBlock] → [CreativeBlock]
                          ↓
               [CuriosityDriver (loop)]

