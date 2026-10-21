# Desafio FSUminho 26/27: Autonomous Systems & Embedded Software (AS / ESW)

> **Duração Total:** 1h30 (90 minutos)  
> **Público-Alvo:** 5 Membros (AS & ESW - Percepção, Controlo, Telemetria e Segurança)  
> **Ambiente:** Python 3 (NumPy, Matplotlib, SciPy, Pandas)

---

## 1. Visão Geral da Missão

O monolugar Driverless da **FSUminho** está pronto para realizar uma sessão de testes em pista fechada (*Trackdrive*). O simulador Software-In-The-Loop (SIL) gera os dados de pista (cones azuis à esquerda, amarelos à direita, laranjas na partida/meta) e simula o feixe do sensor LiDAR/Câmara e a telemetria do CAN Bus (velocidades de roda, acelerador, travão, IMU).

A missão da equipa é implementar os 5 componentes do pipeline em [`desafio_aluno.py`](file:///c:/Users/User/Desktop/FSUminho/26_27/AS/challenge/desafio_aluno.py), validar a estrutura de dados e ver o carro a completar a pista em tempo real no dashboard interativo!

```
                        ┌────────────────────────────────────────┐
                        │   SIMULADOR AO VIVO (CAN / Sensores)   │
                        │    - Cones LiDAR / Câmara com ruído    │
                        │    - Velocidades 4 Rodas, IMU, Pedais  │
                        └───────────────────┬────────────────────┘
                                            │
               ┌────────────────────────────┼────────────────────────────┐
               ▼                            ▼                            ▼
     [AS - PERCEÇÃO]              [AS - CONTROLO]              [ESW - TELEMETRIA]
      Tarefa 1: Filtrar            Tarefa 2: Midline            Tarefa 4: Slip Ratio
      Cones e Ruído                Tarefa 3: Pure Pursuit       Tarefa 5: Safety & EBS
               │                            │                            │
               └────────────────────────────┼────────────────────────────┘
                                            ▼
                        ┌────────────────────────────────────────┐
                        │    DASHBOARD INTERATIVO EM TEMPO REAL  │
                        │   `python simulador_live.py`           │
                        └────────────────────────────────────────┘
```

---

## 2. Divisão de Funções (5 Participantes)

| Membro | Subárea | Tarefa | Ficheiro / Função |
| :--- | :--- | :--- | :--- |
| **Membro 1** | AS (Perceção) | **Tarefa 1:** Filtrar ruído de sensor e separar cones esquerdos e direitos | `filtrar_e_classificar_cones()` |
| **Membro 2** | AS (Planeamento) | **Tarefa 2:** Reconstruir o corredor central (*Midline*) | `calcular_midline_trajetoria()` |
| **Membro 3** | AS (Controlo) | **Tarefa 3:** Implementar o controlador de direção *Pure Pursuit* | `controlador_pure_pursuit()` |
| **Membro 4** | ESW (Telemetria) | **Tarefa 4:** Análise de escorregamento (*Wheel Spin* e *Wheel Lock*) | `analisar_dinamica_veiculo()` |
| **Membro 5** | ESW (Safety) | **Tarefa 5:** Máquina de Estados Driverless e *EBS Watchdog* | `safety_state_machine_ebs()` |

---

## 3. Descrição Técnica das 5 Tarefas

### Tarefa 1: Filtragem e Classificação de Cones (Perceção)
* **Entrada:** Lista de deteções brutas: `[{'x': float, 'y': float, 'color': str, 'confidence': float}, ...]`.
* **Regras:**
  1. Descartar deteções com `confidence < 0.60`.
  2. Ignorar cones com `color == 'unknown'`.
  3. Devolver tuplo `(cones_azuis, cones_amarelos)` com formato `numpy.ndarray` de shape `(N, 2)` e `(M, 2)`.

### Tarefa 2: Extração da Midline (Planeamento)
* **Entrada:** Arrays `cones_azuis` e `cones_amarelos`.
* **Fórmula / Lógica:**
  - Para cada cone azul, encontrar o cone amarelo mais próximo cuja distância esteja entre $2.0\,\text{m}$ e $5.5\,\text{m}$.
  - O ponto central é a média das coordenadas:

$$
\mathbf{P}_{\text{mid}} = \frac{\mathbf{P}_{\text{azul}} + \mathbf{P}_{\text{amarelo}}}{2}
$$

* **Retorno:** `numpy.ndarray` de shape `(K, 2)`.

### Tarefa 3: Pure Pursuit Controller (Controlo)
* **Entrada:** `car_state` $(x, y, \psi, v, \delta)$, `waypoints`, `lookahead_distance` ($L_d = 4.2\,\text{m}$), `wheelbase` ($L = 1.55\,\text{m}$).
* **Fórmula Pure Pursuit:**
  1. Identificar o ponto alvo $\mathbf{P}_t = (x_t, y_t)$ à distância $L_d$ à frente do veículo.
  2. Ângulo relativo: $\alpha = \text{atan2}(y_t - y_c, x_t - x_c) - \psi$.
  3. Ângulo de viragem:

$$
\delta = \arctan\left(\frac{2 L \sin(\alpha)}{L_d}\right)
$$

  4. Limitar entre `[-0.50, +0.50]` radianos ($\approx \pm 28^\circ$).

### Tarefa 4: Escorregamento e Dinâmica (Telemetria ESW)
* **Entrada:** Dicionário com `est_car_velocity`, `rl_wheel_speed`, `rr_wheel_speed`, `fl_wheel_speed`, `apps`, `brake_pressure`, `accel_y`.
* **Fórmulas:**
  - $\text{Slip Ratio} = \frac{V_{\text{roda}} - V_{\text{car}}}{V_{\text{car}}}$ (quando $V_{\text{car}} > 1.0\,\text{m/s}$).
  - **Wheel Spin:** $\text{APPS} > 10\%$ E ($\text{Slip}_{\text{RL}} > 0.15$ OU $\text{Slip}_{\text{RR}} > 0.15$).
  - **Wheel Lock:** $\text{Brake} > 5\,\text{bar}$ E ($\text{Slip}_{\text{FL}} < -0.20$ OU $\text{Slip}_{\text{FR}} < -0.20$).
  - **G Lateral:** $\frac{a_y}{9.81}$.

### Tarefa 5: Safety State Machine & EBS (Segurança ESW)
* **Estados FSG:** `AS_READY` $\to$ `AS_DRIVING` $\to$ `AS_EMERGENCY`.
* **Regras de Disparo do EBS:**
  - Se $\text{distância ao centro da pista} > 2.0\,\text{m}$ (saída de pista / *off-track*).
  - Se $\text{número de cones visíveis} == 0$ (perda de perceção).
  - O acionamento de `AS_EMERGENCY` é permanente e trava o veículo imediatamente!

---

## 4. Como Executar e Validar

### Passo 1: Verificar a Estrutura (Validação de Contrato)
Antes de testar na pista, correr a bateria de testes unitários para garantir que o formato de entrada e saída está correto:
```bash
python verificar_estrutura.py
```
Se passarem nos 5 testes, o terminal mostrará:
```
RESULTADO FINAL: 5/5 Tarefas Aprovadas.
>>> PARABENS! A estrutura cumpre 100% dos requisitos de dados.
```

### Passo 2: Ver o Carro a Andar no Simulador ao Vivo
```bash
python simulador_live.py
```
Irá abrir o dashboard com o carro em tempo real, as linhas de feixe do sensor, o corredor estimado, o indicador de volante e o estado de telemetria ESW!

### Modo Organizador / Gabarito:
Para demonstrar o comportamento perfeito ou tirar dúvidas:
```bash
python verificar_estrutura.py --solucao
python simulador_live.py --solucao
```

---

## 5. Cronograma Recomendado (90 Minutos)

* **00m – 10m:** Apresentação do desafio, divisão dos papéis pelos 5 membros e arranque do simulador.
* **10m – 30m:** Resolução da **Tarefa 1 (Perceção)** e **Tarefa 4 (Telemetria)**.
* **30m – 55m:** Resolução da **Tarefa 2 (Midline)** e **Tarefa 5 (Safety & EBS)**.
* **55m – 75m:** Resolução da **Tarefa 3 (Pure Pursuit)** e afinação no `verificar_estrutura.py`.
* **75m – 85m:** Teste em pista no `simulador_live.py` (completar 2 voltas sem derrubar cones e sem disparos falsos de EBS).
* **85m – 90m:** Conclusão, anúncio de tempos de volta e comparação com a solução de referência.
