# DESAFIO TÉCNICO FSUMINHO 26/27: AUTONOMOUS SYSTEMS & EMBEDDED SOFTWARE (AS / ESW)

> **Missão:** Software-in-the-Loop (SIL) Trackdrive & Vehicle Telemetry
> **Duração:** 1 hora e 30 minutos (90 minutos)
> **Equipa:** 5 Elementos
> **Ficheiro de Trabalho:** [`candidatos/desafio_aluno.py`](../candidatos/desafio_aluno.py)

---

## 1. Contexto e Apresentação do Desafio

O protótipo Driverless da **FSUminho** está na linha de partida para a prova dinâmica de **Trackdrive** da Formula Student. O carro é equipado com sensores LiDAR e câmaras frontais (perceção de cones), encoders de roda, sensor de ângulo de volante, pedal de aceleração (APPS), linha de pressão de travão hidráulico e uma unidade de medição inercial (IMU 6-Axis).

A vossa equipa de 5 elementos é responsável por colocar o monolugar a completar voltas ao circuito de forma autónoma, segura e rápida.

A pista é delimitada por:

* **Cones Azuis:** Limite esquerdo da pista.
* **Cones Amarelos:** Limite direito da pista.
* **Cones Grandes Laranjas:** Linha de Partida / Meta.

---

## 2. Dinâmica de Grupo e Divisão de Papéis

> **Metodologia de Grupo:** As tarefas deste desafio **são resolvidas em equipa colaborativa**. O grupo deve trabalhar de forma integrada: discutir em conjunto as decisões de arquitetura, realizar *brainstorming* e *pair programming*, assegurando que todas as interfaces e fluxos de dados funcionam em perfeita harmonia.

| Tarefa             | Subárea                      | Responsabilidade Principal                                                                       |
| :----------------- | :---------------------------- | :----------------------------------------------------------------------------------------------- |
| **Tarefa 1** | **AS - Perceção**     | Filtrar ruído de sensor, descartar reflexões espúrias e separar cones esquerdos e direitos.   |
| **Tarefa 2** | **AS - Planeamento**    | Reconstruir o corredor de pista e calcular a trajetória central (*Midline*).                  |
| **Tarefa 3** | **AS - Controlo**       | Implementar o controlador cinemático *Pure Pursuit* para direção do veículo.               |
| **Tarefa 4** | **ESW - Telemetria**    | Supervisionar o escorregamento das 4 rodas (*Slip Ratio*) e detetar *Wheel Spin* / *Lock*. |
| **Tarefa 5** | **ESW - Safety & EBS**  | Implementar a Máquina de Estados Driverless e o Watchdog do *Emergency Braking System*.       |
| *Bonus*          | **ESW - Sensor Fusion** | Deteção de desagregação/divergência entre Odometria e GPS com Fail-Safe de EBS.             |

---

## 3. Especificação das Tarefas (5 Obrigatórias + 1 Bónus Opcional)

Todas as funções devem ser implementadas exclusivamente no ficheiro [`candidatos/desafio_aluno.py`](../candidatos/desafio_aluno.py).

---

### Tarefa 1: Filtragem, Classificação e Vetorização de Cones (Perceção)

**Função:** `filtrar_e_classificar_cones(detections: List[Dict]) -> Tuple[np.ndarray, np.ndarray]`

O sensor LiDAR/Câmara devolve uma lista de deteções brutas com coordenadas relativas e probabilidades:

```python
detections = [
    {'x': 3.5, 'y':  1.6, 'color': 'blue',    'confidence': 0.92}, # Cone azul válido
    {'x': 4.1, 'y': -1.5, 'color': 'yellow',  'confidence': 0.85}, # Cone amarelo válido
    {'x': 8.0, 'y':  4.8, 'color': 'unknown', 'confidence': 0.89}, # Artefacto físico exterior (fardo/poste fora da pista)
    {'x': 6.2, 'y':  0.4, 'color': 'blue',    'confidence': 0.35}, # Ruído de sensor LiDAR / poeira / flare
]
```

**Requisitos:**

1. Descartar deteções com `confidence < 0.60` (elimina ruído de sensor espúrio).
2. Descartar cones cuja cor não seja `'blue'` ou `'yellow'` (elimina artefactos físicos fora da pista como fardos de palha ou postes que têm alta confiança geométrica mas não são cones de traçado).
3. Retornar um tuplo `(cones_azuis, cones_amarelos)` onde cada um é um `np.ndarray` de shape `(N, 2)` com as coordenadas `[x, y]`.
4. **Ordenação por Proximidade:**
   - Cada array (`cones_azuis` e `cones_amarelos`) deve vir obrigatoriamente ordenado por ordem crescente de distância à origem ($\sqrt{x^2 + y^2}$), do cone mais próximo ao mais distante.
5. Se um dos lados não tiver cones válidos, deve retornar `np.empty((0, 2))`.

---

### Tarefa 2: Extração do Caminho Central / Midline (Planeamento)

**Função:** `calcular_midline_trajetoria(cones_azuis: np.ndarray, cones_amarelos: np.ndarray) -> np.ndarray`

A partir dos cones azuis e amarelos visíveis, reconstruir os waypoints por onde o monolugar deve guiar.

**Requisitos & Desafio:**

1. A partir dos cones azuis (limite esquerdo) e amarelos (limite direito), devem reconstruir uma trajetória central plausível e contínua.
2. O vosso algoritmo deve lidar de forma autónoma com casos limite:
   - **Emparelhamento Válido:** Nem todos os cones que o sensor vê formam um par de pista legítimo (ex: cones de retas paralelas ou de setores opostos). Devem validar se o par faz sentido para a geometria regulamentar de pista.
   - **Curvas Cegas:** Numa curva fechada, o sensor pode perder temporariamente a visão de um dos lados de cones. O vosso algoritmo deve ser resiliente e manter uma estimativa do traçado em vez de colapsar.
3. Retornar um array `np.ndarray` de shape `(K, 2)` com as coordenadas ordenadas dos waypoints. Se não houver cones suficientes para uma trajetória segura, retornar `np.empty((0, 2))`.

---

### Tarefa 3: Controlador Pure Pursuit & Pedais (Controlo Lateral & Longitudinal)

**Função:** `controlador_pure_pursuit(car_state: CarState, waypoints: np.ndarray, lookahead_distance: float = 4.0, wheelbase: float = 1.55) -> Tuple[float, float, float]`

Determina os comandos de atuação do monolugar: o ângulo de direção das rodas dianteiras ($\delta$, em radianos) e os comandos de pedais de acelerador (`apps_percent`, em %) e travão (`brake_bar`, em bar).

**Requisitos e Fórmulas:**

1. **Ponto de Mira (*Lookahead Point*):**
   Localizar o waypoint da midline situado à distância mais próxima de `lookahead_distance` **à frente do veículo** (no sentido do vetor rumo $\psi$).
2. **Ângulo de Erro $\alpha$:**
   Calcular o ângulo entre a orientação atual do carro ($\psi$) e o ponto alvo:

$$
\alpha = \text{atan2}(y_{\text{alvo}} - y_{\text{car}}, x_{\text{alvo}} - x_{\text{car}}) - \psi \quad \in [-\pi, \pi]
$$

3. **Controlo Lateral (Ângulo de Viragem $\delta$):**

$$
\delta = \arctan\left(\frac{2 \cdot L \cdot \sin(\alpha)}{L_d}\right)
$$

   Onde $L = 1.55\text{ m}$ (*wheelbase*) e $L_d$ é a distância de mira real até ao ponto alvo.
   
   - **Saturação Física:** O atuador de direção só opera no intervalo `[-0.50, +0.50] rad` ($\approx \pm 28^\circ$). Devem limitar (*clamp*) o valor retornado.

4. **Controlo Longitudinal Adaptativo (Acelerador & Travão):**
   O veículo deve gerir a sua velocidade de forma autónoma e suave ao longo do circuito:
   - **Limite Físico do Monolugar:** O monolugar possui um limite físico construtivo de velocidade de **80 km/h** ($\approx 22.2\text{ m/s}$). **Não há qualquer limite artificial ou cap de software imposto no vosso algoritmo** — a equipa tem total liberdade para definir o ritmo e a velocidade alvo que considerar mais competitiva, desde que garanta que o carro não se despista nem sai dos limites de pista!
   - Nas retas (pouca ou nenhuma viragem), o monolugar pode acelerar a fundo para ganhar ritmo (`apps_percent` entre 0.0% e 100.0%).
   - À aproximação e contorno de curvas acentuadas, deve modular a velocidade (aliviando o acelerador e/ou aplicando pressão de travagem `brake_bar` entre 0.0 e 80.0 bar) para garantir a estabilidade do veículo e evitar saídas de pista.
   - A modulação de pedais deve ser progressiva e contínua para evitar esticões bruscos ou oscilações de controlo (*chattering* on/off).
   - *Nota de Engenharia FS:* Evitem acionar acelerador e travão em simultâneo com valores elevados (regra de plausibilidade de pedais da Formula Student).

**Retorno:**
Retornar um tuplo `(steering_angle, apps_percent, brake_bar)` com os três comandos calculados.

---

### Tarefa 4: Telemetria e Análise de Dinâmica (ESW)

**Função:** `analisar_dinamica_veiculo(telemetria: Dict[str, float]) -> Dict[str, Any]`

Supervisionar os sensores de roda e IMU enviados pelo CAN bus.

**Requisitos:**

1. **Slip Ratio Longitudinal:** Para as rodas traseiras motrizes (RL e RR):
   - **Fórmula:** `Slip = (V_wheel - V_car) / V_car`
   - **Condição:** Apenas calcular se `V_car > 1.0 m/s`, caso contrário `Slip = 0.0`.
2. **Deteção de Wheel Spin (Patinagem na Aceleração):**
   Verdadeiro se `APPS > 10%` E (`Slip_RL > 0.15` OU `Slip_RR > 0.15`).
3. **Deteção de Wheel Lock com Filtro de Debounce Temporal (Segurança ESW):**
   - **Condição Instantânea:** Verdadeiro se `Brake > 5 bar` E (`Slip_FL < -0.20` OU `Slip_FR < -0.20`).
   - **Filtro Anti-Ruído Temporal (Debounce):** Para evitar falsos alarmes causados por ruídos pontuais de encoders (ex: ressaltos em zebras ou perda momentânea de sinal), o alarme apenas deve ser confirmado se a condição ocorrer durante **pelo menos 3 amostras consecutivas de tempo** (~0.1 s a 30 Hz). Se a condição deixar de se verificar, o contador reinicia a zero.
4. **Aceleração Lateral em G:** `accel_y / 9.81`.
5. **Geração de Lista de Avisos (`'avisos'`):**
   O algoritmo deve gerar uma lista de avisos para validação em tempo real no simulador:
   - Se houver Wheel Spin: incluir `'AVISO_WHEEL_SPIN'`.
   - Se houver Wheel Lock: incluir `'AVISO_WHEEL_LOCK'`.
   - Se `abs(accel_y / 9.81) > 1.4`: incluir `'AVISO_HIGH_LATERAL_G'`.
   - Se estiver tudo estável: lista vazia `[]`.
6. **Retorno obrigatório:** Dicionário com chaves `'rl_slip'`, `'rr_slip'`, `'wheel_spin_detected'`, `'wheel_lock_detected'`, `'lateral_g'` e `'avisos'`.

---

### Tarefa 5: Safety State Machine & EBS Watchdog (Segurança ESW)

**Função:** `safety_state_machine_ebs(current_state: str, distance_to_midline: float, cones_seen: int, mission_start_cmd: bool = False) -> Tuple[str, bool]`

Garante a conformidade com as regras de segurança Driverless da Formula Student (Rules FSG DV).

**Estados:** `"AS_OFF"`, `"AS_READY"`, `"AS_DRIVING"`, `"AS_EMERGENCY"`.

**Regras de Transição:**

1. Se `current_state == "AS_EMERGENCY"`: Permanece bloqueado em emergência (`"AS_EMERGENCY"`, `True`).
2. Se `current_state == "AS_READY"` e `mission_start_cmd == True`: Transita para (`"AS_DRIVING"`, `False`).
3. Se `current_state == "AS_DRIVING"`:
   - Se `distance_to_midline > 2.0 m` (**Off-Track / Saída de Pista**) OU `cones_seen == 0` (**Perda de Perceção de Pista**):
     - **Disparo Imediato do EBS:** Transita para `("AS_EMERGENCY", True)`.
   - Caso contrário: Mantém `("AS_DRIVING", False)`.

---

### Tarefa Bónus [Opcional]: Supervisão de Fusão Sensorial & Fail-Safe GPS (ESW)

**Função:** `supervisionar_fusao_gps(pos_estimada: Tuple[float, float], pos_gps: Tuple[float, float], dt: float = 0.05) -> Dict[str, Any]`

> **Aviso de Pontuação:** Esta tarefa é **100% opcional**. Caso a equipa não a implemente, continua a poder atingir a nota máxima de 100% (5/5 tarefas). A sua resolução correta concede uma **bonificação extra de +10% (+10 pontos)** na pontuação geral!

Num veículo de competição Formula Student Driverless, a integridade da localização é crítica. O veículo cruza a odometria interna estimada (sensores de roda / SLAM) com as leituras absolutas da antena GNSS/GPS de alta precisão. Se as duas fontes divergirem de forma acentuada e persistente, os sistemas de controlo e perceção deixam de ser confiáveis, exigindo a desativação segura e o disparo do EBS (*Emergency Braking System*).

**Requisitos e Limiares Regulamentares:**

1. **Erro de Distância Euclidiana:**
   Calcular a distância entre a posição estimada $(x_{\text{est}}, y_{\text{est}})$ e a leitura do GPS $(x_{\text{gps}}, y_{\text{gps}})$:

$$
\text{dist} = \sqrt{(x_{\text{est}} - x_{\text{gps}})^2 + (y_{\text{est}} - y_{\text{gps}})^2}
$$

2. **Consenso vs Desagregação (Limiar Espacial de 1.5 m):**
   - Se `erro_distancia <= 1.5 m`: Os sensores estão em consenso. A variável de persistência temporal `_tempo_divergencia_gps` deve ser reiniciada a `0.0`.
   - Se `erro_distancia > 1.5 m`: Os sensores estão desagregados/em divergência. Acumular o passo de tempo: `_tempo_divergencia_gps += dt`.
3. **Persistência Temporal e Gatilhos de Segurança (Limiar Temporal de 1.0 s):**
   - **Caso Normal (`_tempo_divergencia_gps == 0.0`):**
     - `estado_fusao = "FUSION_OK"`
     - `ebs_trigger = False`
     - `avisos = []`
   - **Divergência Transitória / Glitch (`0.0 < _tempo_divergencia_gps < 1.0 s`):**
     - `estado_fusao = "RETRY_RESET"`
     - `ebs_trigger = False`
     - `avisos = ["AVISO_GPS_DRIFT"]`
   - **Divergência Crítica Sustentada (`_tempo_divergencia_gps >= 1.0 s`):**
     - `estado_fusao = "SENSOR_FUSION_LOST"`
     - `ebs_trigger = True` *(disparo de emergência por perda de integridade de localização)*
     - `avisos = ["AVISO_FUSION_FAULT"]`
4. **Retorno Obrigatório:**
   Dicionário com as chaves:
   ```python
   {
       'erro_distancia': float,
       'tempo_divergencia': float,
       'estado_fusao': str,
       'ebs_trigger': bool,
       'avisos': List[str]
   }
   ```

---

## 4. Submissão, Git Branches & Pipeline CI/CD

O desafio está integrado com um repositório central com validação contínua (CI/CD):

> **Repositório Oficial:** [https://github.com/pedroreis2468/challenge](https://github.com/pedroreis2468/challenge)

### Fluxo de Trabalho por Branches e Entregas Contínuas:

> ⚠️ **REGRA OBRIGATÓRIA DE GIT:**
>
> 1. Cria a tua branch no início: `git checkout -b <o-teu-nome>`
> 2. **NUNCA faças push direto para a branch `main`!**
> 3. Dá **SEMPRE push para a tua própria branch**: `git push origin <o-teu-nome>`
> 4. Dá commit e push incremental a cada tarefa concluída para acionar o CI/CD e registar o progresso contínuo no repositório.

Cada candidato/equipa deve trabalhar na sua **branch dedicada**, garantindo total rastreabilidade do progresso ao longo da sessão:

* **Entregas Contínuas por Tarefa:** À medida que concluem ou avançam em cada tarefa (respeitando o tempo de cada bloco), devem efetuar `commit` e `push` imediato para a vossa branch (`git push origin <o-teu-nome>`). Não guardem as alterações todas para o fim! As entregas incrementais alimentam a pipeline de CI/CD e comprovam a evolução do vosso trabalho.

1. **Clonar o Repositório:**

   ```bash
   git clone https://github.com/pedroreis2468/challenge.git
   cd challenge
   ```
2. **Criar a tua Branch Própria:**
   *(Usa o teu nome ou identificador de candidato, ex: `candidato-1`, `candidato-2`):*

   ```bash
   git checkout -b candidato-1
   ```
3. **Programar e Testar Localmente:**
   Todo o desenvolvimento é feito exclusivamente em [`candidatos/desafio_aluno.py`](../candidatos/desafio_aluno.py). Podem validar a qualquer momento:

   ```bash
   python tests/verificar_estrutura.py
   python tests/avaliar_estatico.py --etapa 1
   ```
4. **Submeter e Acionar a Pipeline de CI/CD:**
   Sempre que quiserem testar ou guardar progresso, façam push da vossa branch:

   ```bash
   git add candidatos/desafio_aluno.py
   git commit -m "feat: resolucao da tarefa 1"
   git push origin candidato-1
   ```
5. **Acompanhamento no GitHub Actions:**

   - A pipeline de CI/CD corre **automaticamente a cada push** na vossa branch.
   - Podem acompanhar o resultado no separador **"Actions"** do GitHub (visto verde ✔️ se passar nos testes, ou ❌ com o log de erros).
6. **Declaração Obrigatória de Entregáveis no `README.md` (Mínimo de 3 Tarefas por Candidato):**

   > ⚠️ **Requisito Mínimo Obrigatório:** Cada um deve submeter e assinalar **pelo menos 3 tarefas** concluídas na sua branch para que a submissão individual seja elegível para avaliação. Submissões com menos de 3 tarefas entregues não serão classificadas na prova.
   >

   Nos últimos 5 minutos da sessão, cada um deve editar o ficheiro `README.md` na raiz da sua branch e preencher a secção **"🎯 Declaração de Entregáveis para Avaliação"**, assinalando explicitamente quais as tarefas em que pretende ser avaliado:

   ```markdown
   ## 🎯 Declaração de Entregáveis para Avaliação
   Assinalem com [X] as tarefas concluídas a submeter para avaliação:
   - [X] Tarefa 1: Filtragem e Classificação de Cones (Perceção)
   - [X] Tarefa 2: Trajetória Central / Midline (Planeamento)
   - [X] Tarefa 3: Pure Pursuit & Pedais (Controlo)
   - [X] Tarefa 4: Telemetria & Dinâmica Veicular (ESW)
   - [X] Tarefa 5: Safety State Machine & EBS Watchdog (ESW)
   - [ ] Tarefa Bónus 6: Fusão Sensorial GPS vs Odometria (Opcional)

   ### Notas da Equipa para a Avaliação:
   - (Ex: "Módulos testados no simulador com volta limpa e sem infrações.")
   ```

   *Nota: A equipa avaliadora apenas testará formalmente os módulos que tiverem sido explicitamente assinalados por cada um no seu `README.md`.*
7. **Demonstração Final em Pista (Showcase no Projetor Central):**
   No final do tempo, a organização carregará as branches submetidas no computador central para a demonstração e avaliação dinâmica em projetor através do simulador SIL em tempo real.

---

## 5. Como Executar e Testar Localmente

### 1º Passo: Validador de Estrutura de Dados
Antes de qualquer submissão, executem o verificador na linha de comandos:

```bash
python tests/verificar_estrutura.py
```

O script testará se as vossas funções cumprem o contrato de tipos, dimensões e regras lógicas:

```
RESULTADO FINAL: 5/5 Tarefas Aprovadas.
>>> PARABENS! A estrutura cumpre 100% dos requisitos de dados.
```

### 2º Passo: Avaliação Visual Estática por Etapa
Podem inspecionar graficamente o resultado e os gráficos de cada uma das vossas funções:

```bash
python tests/avaliar_estatico.py --etapa 1  # Etapas de 1 a 6
```
*(Nota: A demonstração dinâmica contínua no simulador em tempo real será realizada e projetada exclusivamente pelo computador central da organização no final da sessão).*

---

## 6. Cronograma & Regra dos 50/50 de Inteligência Artificial

O desafio tem a duração total de **1 hora e 30 minutos (90 minutos)** e segue um modelo de gestão de tempo estruturado por questão, desenhado para avaliar tanto o vosso raciocínio analítico puro como a vossa capacidade de alavancar ferramentas modernas de engenharia.

### ⚖️ A Regra dos 50/50 de IA e os Dois Pushes Obrigatórios

Para cada uma das questões/tarefas, o tempo alocado é dividido rigorosamente em **duas metades**, com **dois commits/pushes obrigatórios por tarefa**:

1. **1ª Metade do Tempo — SEM Inteligência Artificial (🧠 Raciocínio Próprio & Documentação Oficial):**
   - **É estritamente proibido** o uso de assistentes de IA (ChatGPT, Claude, GitHub Copilot, Gemini, Cursor, etc.).
   - A equipa foca-se na interpretação analítica, cálculo de fórmulas no papel/quadro e documentação técnica oficial (Python, NumPy, regras FSG).
   - 📌 **1º Push Obrigatório (Fim da Metade SEM IA):**
     Ao esgotar o tempo da 1ª metade, a equipa comita e envia a versão desenvolvida puramente por si:
     ```bash
     git add candidatos/desafio_aluno.py
     git commit -m "checkpoint: tarefa X sem IA"
     git push origin <o-teu-nome>
     ```

2. **2ª Metade do Tempo — COM Inteligência Artificial (🤖 Aceleração & Coding Assistants):**
   - É **autorizado e incentivado** o uso livre de ferramentas de IA generativa para acelerar a escrita, otimizar código, refatorar e debugar.
   - 📌 **2º Push Obrigatório (Fim da Metade COM IA):**
     Ao terminar a tarefa com o auxílio da IA, a equipa comita e envia a versão refinada:
     ```bash
     git add candidatos/desafio_aluno.py
     git commit -m "feat: tarefa X com IA"
     git push origin <o-teu-nome>
     ```
   *(Esta metodologia permite aos avaliadores da FSUminho analisar tanto a vossa base analítica individual como a vossa capacidade de tirar o máximo partido de ferramentas modernas de engenharia).*

---

### ⏱️ Tabela de Gestão de Tempo por Tarefa

| Janela Temporal      | Tarefa / Etapa                                                                 |   Duração   | 1ª Metade (SEM IA) & 1º Push                                          | 2ª Metade (COM IA) & 2º Push                                       |
| :------------------- | :----------------------------------------------------------------------------- | :---------: | :-------------------------------------------------------------------- | :----------------------------------------------------------------- |
| **00m – 10m**        | **Briefing & Setup:** Leitura de enunciado e criação da branch                 |   10 min    | 10 min (Sem IA) -> `git checkout -b <branch>`                      | —                                                                  |
| **10m – 25m**        | **Tarefa 1 (Perceção):** Filtro de cones, ruído LiDAR e ordenação              |   15 min    | **7.5 min** -> `git commit -m "checkpoint: t1 sem IA"` & `push`    | **7.5 min** -> `git commit -m "feat: t1 com IA"` & `push`       |
| **25m – 45m**        | **Tarefa 2 (Planeamento):** Midline, emparelhamento e curvas cegas             |   20 min    | **10 min** -> `git commit -m "checkpoint: t2 sem IA"` & `push`     | **10 min** -> `git commit -m "feat: t2 com IA"` & `push`        |
| **45m – 65m**        | **Tarefa 3 (Controlo):** Pure Pursuit cinemático e modulação de pedais         |   20 min    | **10 min** -> `git commit -m "checkpoint: t3 sem IA"` & `push`     | **10 min** -> `git commit -m "feat: t3 com IA"` & `push`        |
| **65m – 77m**        | **Tarefa 4 (Telemetria):** Slip Ratio, debounce e lateral G                    |   12 min    | **6 min** -> `git commit -m "checkpoint: t4 sem IA"` & `push`      | **6 min** -> `git commit -m "feat: t4 com IA"` & `push`         |
| **77m – 85m**        | **Tarefa 5 (Safety & EBS):** Máquina de estados Driverless e Watchdog          |    8 min    | **4 min** -> `git commit -m "checkpoint: t5 sem IA"` & `push`      | **4 min** -> `git commit -m "feat: t5 com IA"` & `push`         |
| *(Paralelo)*         | **Tarefa 6 (Bónus Opcional):** Fusão Sensorial GPS vs Odometria & Fail-Safe    | *(paralelo)*| **5 min** -> `git commit -m "checkpoint: bonus sem IA"` & `push`   | **5 min** -> `git commit -m "feat: bonus com IA"` & `push`      |
| **85m – 90m**        | **Declaração README & Fecho:** Preenchimento de entregáveis no `README.md`     |    5 min    | Livre                                                                 | `git commit -m "docs: entregaveis avaliacao"` & `push`             |
