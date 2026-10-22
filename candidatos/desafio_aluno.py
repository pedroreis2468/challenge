"""
=============================================================================
DESAFIO FSUMINHO 26/27: AUTONOMOUS SYSTEMS & EMBEDDED SOFTWARE (AS / ESW)
Tempo Limite: 1h30 | 5 Membros de Equipa
=============================================================================
Instruções:
- Devem implementar as 5 funções abaixo.
- Podem testar a conformidade de dados correndo: `python tests/verificar_estrutura.py`
- Podem gerar e inspecionar os relatórios visuais estáticos por etapa:
  `python tests/avaliar_estatico.py --etapa 1` (a 6)
=============================================================================
"""

import math

import numpy as np
from typing import List, Dict, Tuple, Any
import sys
import os

# Configuração automática de caminhos para a arquitetura do projeto
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
for _path in [
    _ROOT_DIR,
    os.path.join(_ROOT_DIR, "candidatos"),
    os.path.join(_ROOT_DIR, "core"),
    os.path.join(_ROOT_DIR, "organizador"),
    _CURRENT_DIR,
]:
    if os.path.exists(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from contrato_dados import CarState, ASState


# =============================================================================
# TAREFA 1 [AS - PERCEÇÃO]: Filtragem, Separação e Vetorização de Cones
# =============================================================================
def filtrar_e_classificar_cones(
    detections: List[Dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Objetivo:
    Receber uma lista de deteções de cones em formato de dicionários brutos:
      [{'x': float, 'y': float, 'color': 'blue'|'yellow'|'orange'|'unknown', 'confidence': float}, ...]

    Regras de Filtragem e Vetorização:
    1. Descartar deteções com 'confidence' inferior a 0.60 (ruído de sensor / falsos positivos).
    2. Ignorar cones que não sejam das cores da pista ('blue' e 'yellow'), descartando 'unknown'.
    3. Separar os cones válidos em dois arrays numpy de shape (N, 2):
       - cones_azuis: Cones do limite esquerdo da pista [[x, y], ...]
       - cones_amarelos: Cones do limite direito da pista [[x, y], ...]
    4. Ordenação por Proximidade:
       - Cada array (azuis e amarelos) DEVE vir obrigatoriamente ordenado por ordem crescente
         de distância à origem (sensor/veículo, sqrt(x^2 + y^2)), do mais próximo ao mais distante.

    Retorno:
    - (cones_azuis, cones_amarelos): tuplo com dois arrays numpy de shape (N, 2) e (M, 2).
      Se não houver cones de uma das cores, retornar np.empty((0, 2)).
    """
    # TODO: Implementar a filtragem, separação e ordenação vetorizada de cones

    cones_azuis = []
    cones_amarelos = []

    for detection in detections:
        if detection.get("confidence", 0.0) < 0.60:
            continue

        color = detection.get("color")
        if color not in ("blue", "yellow"):
            continue

        try:
            x = float(detection["x"])
            y = float(detection["y"])
        except (KeyError, TypeError, ValueError):
            continue

        if color == "blue":
            cones_azuis.append([x, y])
        else:
            cones_amarelos.append([x, y])

    def ordenar_por_distancia(cones):
        cones.sort(key=lambda point: math.hypot(point[0], point[1]))
        if not cones:
            return np.empty((0, 2), dtype=float)
        return np.asarray(cones, dtype=float).reshape(-1, 2)

    return ordenar_por_distancia(cones_azuis), ordenar_por_distancia(cones_amarelos)


# =============================================================================
# TAREFA 2 [AS - PLANEAMENTO]: Extração do Caminho Central (Midline)
# =============================================================================
def calcular_midline_trajetoria(
    cones_azuis: np.ndarray, cones_amarelos: np.ndarray
) -> np.ndarray:
    """
    Objetivo:
    A partir das coordenadas dos cones azuis (esquerda) e amarelos (direita),
    estimar a linha central ideal (midline waypoints) por onde o carro deve circular.

    Entrada:
    - cones_azuis: numpy array de shape (N, 2)
    - cones_amarelos: numpy array de shape (M, 2)

    Regras / Desafio:
    1. A partir dos cones azuis e amarelos visíveis, devem reconstruir uma sequência de
       pontos centrais (midline) ordenada no sentido de progressão da pista.
    2. Atenção aos casos limite (edge cases):
       - O que acontece se a distância entre um par for fisicamente impossível para a largura da pista?
       - O que fazer se numa curva apertada o sensor apenas detetar cones de um dos lados?
    3. Se não houver cones suficientes para uma estimativa segura: retornar np.empty((0, 2)).

    Retorno:
    - waypoints: numpy array de shape (K, 2) contendo a lista ordenada de pontos centrais [[x, y], ...]
    """
    # TODO: Implementar o cálculo da trajetória central (midline)
    raise NotImplementedError("Tarefa 2 por implementar!")


# =============================================================================
# TAREFA 3 [AS - CONTROLO]: Algoritmo de Direção Pure Pursuit & Controlo Longitudinal
# =============================================================================

def _get_state_value(car_state: Any, names: Tuple[str, ...], default: float = 0.0) -> float:
    """Lê o primeiro atributo existente, permitindo nomes comuns no estado do carro."""
    for name in names:
        if hasattr(car_state, name):
            try:
                return float(getattr(car_state, name))
            except (TypeError, ValueError):
                pass
    return default

def controlador_pure_pursuit(
    car_state: CarState,
    waypoints: np.ndarray,
    lookahead_distance: float = 4.0,
    wheelbase: float = 1.55,
) -> Tuple[float, float, float]:
    """
    Objetivo:
    Calcular tanto a direção (Pure Pursuit) como a aceleração/travagem (Controlo Longitudinal)
    para o monolugar percorrer a pista de forma rápida, estável e sem sair dos limites.

    Entrada:
    - car_state: Objeto CarState com (x, y, yaw, v, steer)
    - waypoints: Array numpy (K, 2) da trajetória central calculada na Tarefa 2.
    - lookahead_distance: Distância de mira para o waypoint alvo (default 4.0 metros).
    - wheelbase: Distância entre eixos do monolugar FSUminho (1.55 metros).

    Algoritmo & Desafio:
    1. Controlo Lateral (Direção):
       - Identificar o waypoint alvo situado à distância de mira à frente do veículo.
       - Calcular o desvio angular (alpha) e o ângulo de esterçamento (delta) via Pure Pursuit.
       - Respeitar a saturação física da cremalheira de direção.
    2. Controlo Longitudinal (Acelerador & Travão):
       - Em retas (volante quase a zero): acelerar forte (APPS %).
       - Em curvas acentuadas ou curvatura alta: aliviar o acelerador ou aplicar pressão de travão
         para evitar perdas de aderência ou subviragem (understeer).

    Retorno:
    - (steering_angle, apps_percent, brake_bar):
      * steering_angle: float (em radianos)
      * apps_percent: float entre 0.0% e 100.0%
      * brake_bar: float entre 0.0 bar e 80.0 bar
    """
    # TODO: Implementar o Controlo Lateral e Longitudinal do monolugar

    points = np.asarray(waypoints, dtype=float).reshape(-1, 2)
    if len(points) == 0:
        return 0.0, 0.0, 80.0

    x_car = _get_state_value(car_state, ("x", "x_m", "pos_x"))
    y_car = _get_state_value(car_state, ("y", "y_m", "pos_y"))
    heading = _get_state_value(car_state, ("heading", "yaw", "psi", "yaw_angle"))
    speed = _get_state_value(car_state, ("speed", "velocity", "speed_mps", "v"))

    # Só considerar pontos à frente, usando a projeção no vetor de orientação.
    dx = points[:, 0] - x_car
    dy = points[:, 1] - y_car
    forward = dx * math.cos(heading) + dy * math.sin(heading)
    ahead_indices = np.where(forward > 0.0)[0]

    if len(ahead_indices) == 0:
        return 0.0, 0.0, 80.0

    distances = np.hypot(dx[ahead_indices], dy[ahead_indices])
    target_index = ahead_indices[int(np.argmin(np.abs(distances - lookahead_distance)))]
    target_x, target_y = points[target_index]

    alpha = math.atan2(target_y - y_car, target_x - x_car) - heading
    alpha = math.atan2(math.sin(alpha), math.cos(alpha))
    target_distance = max(float(math.hypot(target_x - x_car, target_y - y_car)), 0.1)

    steering = math.atan2(2.0 * wheelbase * math.sin(alpha), target_distance)
    steering = max(-0.50, min(0.50, steering))

    # Alvo de velocidade mais baixo quando a direção exigida é maior.
    target_speed = 22.2 - 12.0 * min(abs(steering) / 0.50, 1.0)
    target_speed = max(6.0, target_speed)

    if speed < target_speed - 0.5:
        throttle = min(100.0, max(0.0, (target_speed - speed) * 12.0))
        brake = 0.0
    elif speed > target_speed + 0.5:
        throttle = 0.0
        brake = min(80.0, (speed - target_speed) * 8.0)
    else:
        throttle = 15.0
        brake = 0.0

    return float(steering), float(throttle), float(brake)


# Estado persistente para filtro temporal (debounce de Wheel Lock)
_wheel_lock_consecutive_count = 0


# =============================================================================
# TAREFA 4 [ESW - TELEMETRIA & DINÂMICA]: Análise de Escorregamento (Slip Ratio)
# =============================================================================
def analisar_dinamica_veiculo(telemetria: Dict[str, float]) -> Dict[str, Any]:
    """
    Objetivo:
    Supervisionar os sensores de roda e IMU para garantir estabilidade e tração.

    Entrada: Dicionário de telemetria contendo:
      - 'est_car_velocity': float (m/s)
      - 'fl_wheel_speed': float (m/s)
      - 'fr_wheel_speed': float (m/s)
      - 'rl_wheel_speed': float (m/s)
      - 'rr_wheel_speed': float (m/s)
      - 'accel_x': float (m/s^2)
      - 'accel_y': float (m/s^2)
      - 'apps': float (0 a 100%)
      - 'brake_pressure': float (0 a 80 bar)

    Cálculos a realizar:
    1. Calcular o Slip Ratio das rodas traseiras motrizes (RL e RR):
       Se est_car_velocity > 1.0 m/s:
          slip = (v_wheel - v_car) / v_car
       Senão:
          slip = 0.0
    2. Deteção de Wheel Spin (Patinagem na Aceleração):
       Verdadeiro se (apps > 10%) E (RL_slip > 0.15 OU RR_slip > 0.15).
    3. Deteção de Wheel Lock (Bloqueio das Dianteiras com Filtro de Debounce Temporal):
       - Condição instantânea: (brake_pressure > 5 bar) E (FL_slip < -0.20 OU FR_slip < -0.20).
       - Filtro Anti-Ruído (Debounce): Para evitar falsos positivos causados por ruídos
         momentâneos de encoders (zebras, ressaltos), o alarme apenas é confirmado
         se a condição ocorrer durante pelo menos 3 AMOSTRAS CONSECUTIVAS (~0.1s a 30Hz).
         Se a condição deixar de se verificar, o contador reinicia a 0.
    4. Aceleração lateral em G: accel_y / 9.81
    5. Lista de Avisos de Telemetria (para o simulador verificar em tempo real):
       - Se detetar Wheel Spin: incluir a string 'AVISO_WHEEL_SPIN'
       - Se detetar Wheel Lock confirmado: incluir a string 'AVISO_WHEEL_LOCK'
       - Se abs(lateral_g) > 1.4: incluir a string 'AVISO_HIGH_LATERAL_G'
       - Se tudo estiver nominal: lista vazia []

    Retorno: Dicionário com a estrutura exata:
      {
         'rl_slip': float,
         'rr_slip': float,
         'wheel_spin_detected': bool,
         'wheel_lock_detected': bool,
         'lateral_g': float,
         'avisos': List[str]
      }
    """
    # TODO: Implementar a análise de telemetria veicular e lista de avisos

    global _wheel_lock_samples

    v_car = telemetria["est_car_velocity"]
    v_fl = telemetria["fl_wheel_speed"]
    v_fr = telemetria["fr_wheel_speed"]
    v_rl = telemetria["rl_wheel_speed"]
    v_rr = telemetria["rr_wheel_speed"]
    apps = telemetria["apps"]
    brake = telemetria["brake_pressure"]
    accel_y = telemetria["accel_y"]

    if v_car > 1.0:
        fl_slip = (v_fl - v_car) / v_car
    else:
        fl_slip = 0.0

    if v_car > 1.0:
        fr_slip = (v_fr - v_car) / v_car
    else:
        fr_slip = 0.0

    if v_car > 1.0:
        rl_slip = (v_rl - v_car) / v_car
    else:
        rl_slip = 0.0

    if v_car > 1.0:
        rr_slip = (v_rr - v_car) / v_car
    else:
        rr_slip = 0.0

    wheel_spin = apps > 10.0 and (rl_slip > 0.15 or rr_slip > 0.15)

    wheel_lock_condition = brake > 5.0 and (fl_slip < -0.20 or fr_slip < -0.20)
    if wheel_lock_condition:
        _wheel_lock_samples += 1
    else:
        _wheel_lock_samples = 0
    wheel_lock = _wheel_lock_samples >= 3

    lateral_g = accel_y / 9.81

    avisos = []
    if wheel_spin:
        avisos.append("AVISO_WHEEL_SPIN")
    if wheel_lock:
        avisos.append("AVISO_WHEEL_LOCK")
    if abs(lateral_g) > 1.4:
        avisos.append("AVISO_HIGH_LATERAL_G")

    return {
        "rl_slip": float(rl_slip),
        "rr_slip": float(rr_slip),
        "wheel_spin_detected": bool(wheel_spin),
        "wheel_lock_detected": bool(wheel_lock),
        "lateral_g": float(lateral_g),
        "avisos": avisos,
    }


# =============================================================================
# TAREFA 5 [ESW - SAFETY STATE MACHINE & EBS]: Supervisão de Segurança Driverless
# =============================================================================
def safety_state_machine_ebs(
    current_state: str,
    distance_to_midline: float,
    cones_seen: int,
    mission_start_cmd: bool = False,
) -> Tuple[str, bool]:
    """
    Objetivo:
    Implementar a lógica da Máquina de Estados de Autónomo e do sistema
    de Travagem de Emergência (EBS - Emergency Braking System) segundo as regras FSG.

    Estados Possíveis:
    - "AS_OFF": Carro desligado
    - "AS_READY": Pronto para missão
    - "AS_DRIVING": Carro em andamento autónomo
    - "AS_EMERGENCY": EBS disparado / Emergência ativa (irrecuperável na pista)

    Regras de Transição:
    1. Se current_state == "AS_EMERGENCY":
       - Permanece em "AS_EMERGENCY", e ebs_active = True (não pode sair de emergência).
    2. Se current_state == "AS_READY":
       - Se mission_start_cmd == True: transita para "AS_DRIVING", ebs_active = False.
       - Senão: continua em "AS_READY", ebs_active = False.
    3. Se current_state == "AS_DRIVING":
       - CONDIÇÕES DE DISPARO DO EBS:
         * Se distance_to_midline > 2.0 metros (Off-track: o carro saiu da pista!) OU
         * Se cones_seen == 0 (Perda total de perceção de pista!)
         -> Transita imediatamente para "AS_EMERGENCY" e ativa ebs_active = True!
       - Caso contrário:
         -> Mantém "AS_DRIVING" e ebs_active = False.

    Retorno:
    - (novo_estado, ebs_active): Tuple[str, bool]
      Exemplo: ("AS_DRIVING", False) ou ("AS_EMERGENCY", True)
    """
    # TODO: Implementar a máquina de estados e o EBS
    raise NotImplementedError("Tarefa 5 por implementar!")


# =============================================================================
# TAREFA BÓNUS [OPCIONAL - FUSÃO SENSORIAL & FAIL-SAFE GPS / ODOMETRIA]
# =============================================================================
# Variável global para acumular a persistência temporal de divergência (em segundos)
_tempo_divergencia_gps = 0.0


def supervisionar_fusao_gps(
    pos_estimada: Tuple[float, float], pos_gps: Tuple[float, float], dt: float = 0.05
) -> Dict[str, Any]:
    """
    Objetivo [BÓNUS OPCIONAL]:
    Supervisionar a integridade entre a estimativa interna do carro (pos_estimada da odometria/SLAM)
    e as coordenadas externas fornecidas pela antena GNSS/GPS (pos_gps).

    Limiares Regulamentares FSG:
    1. Erro de Distância Euclidiana:
       dist_err = sqrt((x_est - x_gps)^2 + (y_est - y_gps)^2)
       - Limiar Espacial: 1.5 metros.
       - Se dist_err <= 1.5 m: Sensores em consenso (não há divergência). O contador de tempo reinicia a 0.0.
       - Se dist_err > 1.5 m: Sensores em desagregação. Acumular o tempo: _tempo_divergencia_gps += dt.

    2. Filtragem Temporal de Persistência e Gatilhos de Segurança:
       - Se _tempo_divergencia_gps == 0.0:
           estado_fusao = "FUSION_OK"
           ebs_trigger = False
           avisos = []
       - Se 0.0 < _tempo_divergencia_gps < 1.0 s (Pico/Glitch ou Divergência Transitória):
           estado_fusao = "RETRY_RESET"
           ebs_trigger = False
           avisos = ["AVISO_GPS_DRIFT"]
       - Se _tempo_divergencia_gps >= 1.0 s (Divergência Crítica Sustentada):
           estado_fusao = "SENSOR_FUSION_LOST"
           ebs_trigger = True  # Dispara travagem de emergência por perda de integridade de localização
           avisos = ["AVISO_FUSION_FAULT"]

    Retorno:
    Dicionário com a estrutura exata:
      {
         'erro_distancia': float,
         'tempo_divergencia': float,
         'estado_fusao': str,
         'ebs_trigger': bool,
         'avisos': List[str]
      }
    """
    # TODO: Implementar a supervisão de fusão sensorial (OPCIONAL)
    raise NotImplementedError("Tarefa Bónus por implementar!")
