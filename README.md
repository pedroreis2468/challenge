# 🏎️ DESAFIO TÉCNICO FSUMINHO 26/27

## Autonomous Systems & Embedded Software (AS / ESW)

> **Missão:** Software-in-the-Loop (SIL) Trackdrive & Vehicle Telemetry
> **Duração:** 1h30 (90 minutos)
> **Ficheiro de Trabalho:** [`candidatos/desafio_aluno.py`](candidatos/desafio_aluno.py)
> **Enunciado Completo com Fórmulas:** [`docs/ENUNCIADO_DESAFIO.md`](docs/ENUNCIADO_DESAFIO.md)

---

## 📌 Contexto da Prova

O protótipo autónomo da **FSUminho** está na linha de partida para a prova dinâmica de **Trackdrive** da Formula Student. O veículo é equipado com sensores LiDAR e câmaras frontais (perceção de cones), wheel speed sensors, sensor de ângulo de volante, sensor de posição do acelerador (APPS), sensor da linha de pressão de travagem e IMU.

O vosso objetivo é implementar o pipeline de software em [`candidatos/desafio_aluno.py`](candidatos/desafio_aluno.py) para que o carro complete voltas à pista de forma rápida, autónoma e sem derrubar cones:

* **Cones Azuis:** Limite esquerdo da pista.
* **Cones Amarelos:** Limite direito da pista.
* **Cones Grandes Laranjas:** Linha de partida / meta.

---

## 🌿 Fluxo de Trabalho Git (Branches & CI/CD)

> ⚠️ **REGRA DE OURO DO GIT:**
> 1. Cria a tua branch logo no início com o teu identificador: `git checkout -b <o-teu-nome>`
> 2. **NUNCA faças push para a branch `main`!**
> 3. Dá **SEMPRE push para a tua própria branch**: `git push origin <o-teu-nome>`
> 4. À medida que completares cada questão, dá commit e push para a tua branch para registar o teu progresso e acionar a pipeline de validação automática no GitHub Actions.

Cada participante/equipa deve trabalhar numa **branch dedicada**, permitindo validar o código continuamente no GitHub Actions:

### 1. Clonar o Repositório e Criar Branch

```bash
git clone https://github.com/pedroreis2468/challenge.git
cd challenge

# Cria a tua branch com o teu nome ou identificador
git checkout -b candidato-1
```

### 2. Programar Exclusivamente em `candidatos/desafio_aluno.py`

Todo o código a desenvolver deve ser implementado no ficheiro:
👉 [`candidatos/desafio_aluno.py`](candidatos/desafio_aluno.py)

### 3. Validar Localmente

A qualquer momento podem verificar se a estrutura e as regras matemáticas estão conformes:

```bash
# Validador de contrato rápido (5/5 tarefas)
python tests/verificar_estrutura.py

# Avaliação visual da Etapa 1
python tests/avaliar_estatico.py --etapa 1
```

### 4. Entregas Contínuas por Questão (Sem IA vs Com IA)

Cada questão tem duas fases e **dois pushes obrigatórios** para a vossa branch:

1. **No final da 1ª Metade (SEM IA):**
   ```bash
   git add candidatos/desafio_aluno.py
   git commit -m "checkpoint: tarefa X sem IA"
   git push origin <o-teu-nome>
   ```

2. **No final da 2ª Metade (COM IA):**
   ```bash
   git add candidatos/desafio_aluno.py
   git commit -m "feat: tarefa X com IA"
   git push origin <o-teu-nome>
   ```

Acompanhem o resultado no separador **"Actions"** do vosso repositório no GitHub (✔️ se passar nos testes, ❌ se houver erros).

### 5. Declaração Final de Entregáveis no README

Antes do fim dos 90 minutos, cada um deve editar a secção **"🎯 Declaração de Entregáveis para Avaliação"** no final deste ficheiro na sua branch, assinalando com `[X]` as tarefas em que pretende ser formalmente avaliado (**mínimo obrigatório de 3 tarefas por candidato** para a submissão ser aceite para avaliação).

---

## 📋 Resumo das 5 Tarefas

| Tarefa             | Função a Implementar            | Descrição                                                                                                                                            |
| :----------------- | :-------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tarefa 1** | `filtrar_e_classificar_cones()` | Filtrar ruído de sensor (`confidence >= 0.60`), descartar `'unknown'` e ordenar cones por proximidade crescente ($\sqrt{x^2 + y^2}$) via NumPy. |
| **Tarefa 2** | `calcular_midline_trajetoria()` | Reconstruir o corredor central da pista emparelhando pares legítimos e tratar curvas cegas.                                                           |
| **Tarefa 3** | `controlador_pure_pursuit()`    | Calcular ângulo de viragem $\delta$ das rodas dianteiras e modular acelerador (`apps_percent`) e travão (`brake_bar`).                         |
| **Tarefa 4** | `analisar_dinamica_veiculo()`   | Calcular *Slip Ratio* das rodas, detetar eventos de *Wheel Spin* e *Wheel Lock* e acelerar em G.                                                 |
| **Tarefa 5** | `safety_state_machine_ebs()`    | Máquina de Estados Driverless (`AS_READY` $\to$ `AS_DRIVING` $\to$ `AS_EMERGENCY`) e gatilhos de corte imediato do EBS.                     |

*(Consultem [`docs/ENUNCIADO_DESAFIO.md`](docs/ENUNCIADO_DESAFIO.md) para fórmulas matemáticas detalhadas e regras de transição).*

---

## 📊 Como Validar e Testar Localmente

### 1. Validador Rápido de Contrato (CI/CD)
```bash
python tests/verificar_estrutura.py
```

### 2. Relatórios de Avaliação Estática por Etapa
Podem gerar a qualquer momento os relatórios gráficos e diagramas de validação de cada tarefa:
```bash
python tests/avaliar_estatico.py --etapa 1  # (ou --etapa 2 a 6)
```
*(A simulação dinâmica ao vivo será projetada no ecrã central pela organização durante o showcase final).*

---

## 📁 Estrutura do Repositório

```text
challenge/
│
├── 📂 candidatos/             # Pasta dedicada de trabalho dos candidatos
│   └── 🎯 desafio_aluno.py    # ONDE DESENVOLVEM AS 5 TAREFAS
│
├── 📂 core/                   # Motor e contratos do simulador SIL
│   ├── ⚙️ contrato_dados.py   # Classes de dados (CarState, ASState)
│   └── ⚙️ gerador_pista.py    # Gerador procedural de circuitos
│
├── 📂 tests/                  # Bateria de testes e simulação
│   ├── 🧪 verificar_estrutura.py # Validador rápido de conformidade (CI/CD)
│   ├── 🏎️ simulador_live.py     # Simulador SIL em tempo real
│   └── 📊 avaliar_estatico.py   # Avaliador visual estático por etapa
│
├── 📂 docs/                   # Documentação e regras da prova
│   └── 📚 ENUNCIADO_DESAFIO.md# Enunciado completo com fórmulas detalhadas
│
└── 📄 README.md               # Guia rápido de arranque
```

---

## 🎯 Declaração de Entregáveis para Avaliação

> ⚠️ **Requisito Mínimo Obrigatório:** Cada um deve entregar e assinalar **pelo menos 3 tarefas** para a submissão ser considerada elegível para avaliação.
>
> Antes do final do desafio, preencham a lista abaixo assinalando com `[X]` quais as tarefas que concluíram e pretendem que a equipa técnica da FSUminho teste e avalie.

- [X] **Tarefa 1:** Filtragem, Classificação e Vetorização de Cones (Perceção)
- [X] **Tarefa 2:** Extração do Caminho Central / Midline (Planeamento)
- [X] **Tarefa 3:** Controlador Pure Pursuit & Pedais (Controlo Lateral & Longitudinal)
- [ ] **Tarefa 4:** Telemetria e Análise de Dinâmica - Slip & Wheel Spin/Lock (ESW)
- [X] **Tarefa 5:** Safety State Machine & EBS Watchdog (ESW)
- [ ] **Tarefa Bónus 6 (Opcional):** Supervisão de Fusão Sensorial GPS vs Odometria (ESW)

### 📝 Notas Adicionais para os Avaliadores:

*(Espaço para cada um explicar opções de implementação, abordagens seguidas ou casos particulares)*

## Utilização de Inteligência Artificial

Ferramenta utilizada: ChatGPT/Codex.

Utilizei IA para interpretar pseudocódigos, gerar implementações e realizar ajustes ao código

### Tarefa 1 — Filtragem de cones

Apresentei os requisitos de filtragem, separação por cor e ordenação
por distância. Depois solicitei uma versão que reduzisse o processamento.

A IA contribuiu com a implementação em NumPy e com a utilização das
distâncias ao quadrado para evitar raízes quadradas na ordenação.

### Tarefa 2 — Trajetória central

Implementei o pseudocódigo e, posteriormente,
na sua implementação sem variáveis globais.

A IA contribuiu com o emparelhamento geométrico dos cones e com a estimativa da
trajetória quando apenas um dos lados está visível.

### Tarefa 3 — Pure Pursuit e pedais

Apresentei o meu pseudocódigo para determinar a velocidade desejada,
antecipar curvas, controlar acelerador e travão e suavizar os comandos.

Solicitei à IA a análise desse raciocínio e a implementação.
Posteriormente pedi a remoção das variáveis globais.

A IA contribuiu com os cálculos de direção, curvatura, velocidade
desejada, antecipação de travagem e suavização dos pedais.

### Tarefa 5 — Máquina de estados e EBS

Enviei um pseudocódigo inicial à IA que adaptei-o no ficheiro
do projeto. Depois pedi a implementação com base nessa versão,
sem utilizar variáveis globais.

A IA contribuiu com a implementação das transições, a correção
do nome AS_EMERGENCY e a manutenção do EBS desligado durante
a espera em AS_READY.

## Registo dos pedidos

Os pedidos acima estão resumidos por tarefa. Incluem apoio na
elaboração do pseudocódigo, implementação, otimização e correção
do código.

------------------------------------------------------------------------------------------------------

---

Boa sorte na pista! 🏎️💨
