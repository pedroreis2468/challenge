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

import numpy as np
from typing import List, Dict, Tuple, Any
from math import atan, cos, isfinite, sin, sqrt, tan
import sys
import os
# Configuração automática de caminhos para a arquitetura do projeto
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
for _path in [_ROOT_DIR, os.path.join(_ROOT_DIR, "candidatos"), os.path.join(_ROOT_DIR, "core"), os.path.join(_ROOT_DIR, "organizador"), _CURRENT_DIR]:
    if os.path.exists(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from contrato_dados import CarState, ASState

# =============================================================================
# TAREFA 1 [AS - PERCEÇÃO]: Filtragem, Separação e Vetorização de Cones
# =============================================================================
def filtrar_e_classificar_cones(detections: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    # TODO: Implementar a filtragem, separação e ordenação vetorizada de cones
    """Filtra os cones e ordena cada cor por distância crescente à origem."""
    grupos = {"blue": [], "yellow": []}

    for detection in detections:
        if detection["confidence"] < 0.60:
            continue

        grupo = grupos.get(detection["color"])
        if grupo is not None:
            grupo.append((detection["x"], detection["y"]))

    def ordenar(pontos: list) -> np.ndarray:
        if not pontos:
            return np.empty((0, 2), dtype=float)

        cones = np.asarray(pontos, dtype=float)

        if len(cones) > 1:
            # x² + y² mantém a ordem das distâncias sem calcular raízes.
            distancias_quadradas = np.einsum("ij,ij->i", cones, cones)
            cones = cones[np.argsort(distancias_quadradas)]

        return cones

    return ordenar(grupos["blue"]), ordenar(grupos["yellow"])



# =============================================================================
# TAREFA 2 [AS - PLANEAMENTO]: Extração do Caminho Central (Midline)
# =============================================================================
def calcular_midline_trajetoria(cones_azuis: np.ndarray, cones_amarelos: np.ndarray) -> np.ndarray:
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
def controlador_pure_pursuit(car_state: CarState, waypoints: np.ndarray, 
                             lookahead_distance: float = 4.0, wheelbase: float = 1.55) -> Tuple[float, float, float]:
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
    # Parâmetros locais do simulador e valores iniciais de afinação (SI).
    dt = 0.035
    velocidade_maxima = 80.0 / 3.6
    aceleracao_maxima = 4.2
    travagem_maxima = 8.5
    aceleracao_lateral_maxima = 5.0
    travagem_planeada = 4.0
    jerk_subida = 8.0
    jerk_descida = 30.0

    def parar() -> Tuple[float, float, float]:
        car_state._pp_aceleracao_anterior = None
        return 0.0, 0.0, 80.0

    pontos = np.asarray(waypoints, dtype=float)
    if (pontos.ndim != 2 or pontos.shape[1] != 2 or len(pontos) == 0
            or not np.isfinite(pontos).all()):
        return parar()

    if (not all(map(isfinite, (car_state.x, car_state.y, car_state.yaw,
                              car_state.v, car_state.steer,
                              lookahead_distance, wheelbase)))
            or lookahead_distance <= 0.0 or wheelbase <= 0.0):
        return parar()

    # Os waypoints são globais. Projetar apenas o necessário no rumo do carro.
    relativos = pontos - (car_state.x, car_state.y)
    c, s = cos(car_state.yaw), sin(car_state.yaw)
    frente = relativos[:, 0] * c + relativos[:, 1] * s
    indices = np.flatnonzero(frente > 1e-8)
    if indices.size == 0:
        return parar()

    distancias2 = np.einsum("ij,ij->i", relativos, relativos)
    # A proximidade ao lookahead exige distâncias reais, não ao quadrado.
    distancias_frente = np.sqrt(distancias2[indices])
    alvo = indices[np.argmin(np.abs(distancias_frente - lookahead_distance))]
    lateral = -relativos[alvo, 0] * s + relativos[alvo, 1] * c
    curvatura_mira = 2.0 * lateral / distancias2[alvo]
    delta = max(-0.50, min(0.50, atan(wheelbase * curvatura_mira)))

    # a_lateral = v² * |curvatura|. Considerar também o volante ainda aplicado.
    curvatura_atual = max(abs(curvatura_mira),
                         abs(tan(car_state.steer) / wheelbase))
    velocidade2 = velocidade_maxima ** 2
    if curvatura_atual > 1e-8:
        velocidade2 = min(velocidade2, aceleracao_lateral_maxima / curvatura_atual)

    # Preservar a ordem da trajetória, incluindo curvas que saem do semiplano
    # frontal. Remover só repetições consecutivas antes de calcular curvaturas.
    caminho = pontos[indices[0]:]
    segmentos = np.diff(caminho, axis=0)
    manter = np.concatenate(([True], np.any(segmentos != 0.0, axis=1)))
    caminho = caminho[manter]
    if len(caminho) >= 3:
        segmentos = np.diff(caminho, axis=0)
        comprimentos = np.hypot(segmentos[:, 0], segmentos[:, 1])
        a, b = segmentos[:-1], segmentos[1:]
        cordas = caminho[2:] - caminho[:-2]
        denominador = (comprimentos[:-1] * comprimentos[1:]
                       * np.hypot(cordas[:, 0], cordas[:, 1]))
        cruzamento = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        curvaturas = np.divide(2.0 * cruzamento, denominador,
                               out=np.full_like(denominador, np.inf),
                               where=denominador > 1e-12)
        limites2 = np.divide(aceleracao_lateral_maxima, curvaturas,
                             out=np.full_like(curvaturas, velocidade2),
                             where=curvaturas > 1e-8)

        # Distância ao longo do caminho até ao centro de cada trio de pontos.
        distancias_curvas = (sqrt(distancias2[indices[0]])
                             + np.cumsum(comprimentos)[:-1])
        # Margem de 1 m + 0.3 s de avanço para acomodar resposta e suavização.
        margem = 1.0 + 0.3 * max(car_state.v, 0.0)
        distancias_travagem = np.maximum(distancias_curvas - margem, 0.0)
        # v_agora² <= v_curva² + 2 * desaceleração * distância.
        limites2 += 2.0 * travagem_planeada * distancias_travagem
        velocidade2 = min(velocidade2, float(np.min(limites2)))

    erro = sqrt(max(velocidade2, 0.0)) - max(car_state.v, 0.0)
    # Zona morta contínua para evitar alternância dos pedais perto do alvo.
    if erro > 0.15:
        aceleracao = min(aceleracao_maxima, 1.5 * (erro - 0.15))
    elif erro < -0.15:
        aceleracao = max(-travagem_maxima, 2.0 * (erro + 0.15))
    else:
        aceleracao = 0.0

    # Cada veículo guarda a sua memória de suavização no próprio CarState.
    # O simulador reutiliza este objeto entre ciclos.
    anterior = getattr(car_state, "_pp_aceleracao_anterior", None)
    if anterior is not None:
        if anterior * aceleracao < 0.0:
            anterior = 0.0  # Cortar o pedal anterior antes de trocar de sinal.
        variacao = aceleracao - anterior
        aceleracao = anterior + max(-jerk_descida * dt,
                                    min(jerk_subida * dt, variacao))

    car_state._pp_aceleracao_anterior = aceleracao
    acelerador = 100.0 * max(aceleracao, 0.0) / aceleracao_maxima
    travao = 80.0 * max(-aceleracao, 0.0) / travagem_maxima
    return float(delta), float(acelerador), float(travao)


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
    raise NotImplementedError("Tarefa 4 por implementar!")


# =============================================================================
# TAREFA 5 [ESW - SAFETY STATE MACHINE & EBS]: Supervisão de Segurança Driverless
# =============================================================================
def safety_state_machine_ebs(current_state: str, distance_to_midline: float, 
                             cones_seen: int, mission_start_cmd: bool = False) -> Tuple[str, bool]:
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
    
    Pseudocodigo:
    if estadoAtaul == "AAS_EMERGENCY"
        then ("AS_EMERGENCY", True)

    if estadoAtual == "AS_READY":
        if comando_arranque == True:
            then ("AS_DRIVING", False)
        then (AS_READY, True)

    if estado_atual == "AS_DRIVING":
        if distancia_ao_centro > 2.0 OU cones_visiveis == 0:
            then ("AS_EMERGENCY", True)

        then ("AS_DRIVING", False)

    return (estado_atual, False)
    
    """


    # TODO: Implementar a máquina de estados e o EBS
    raise NotImplementedError("Tarefa 5 por implementar!")

# =============================================================================
# TAREFA BÓNUS [OPCIONAL - FUSÃO SENSORIAL & FAIL-SAFE GPS / ODOMETRIA]
# =============================================================================
# Variável global para acumular a persistência temporal de divergência (em segundos)
_tempo_divergencia_gps = 0.0

def supervisionar_fusao_gps(pos_estimada: Tuple[float, float], 
                           pos_gps: Tuple[float, float], 
                           dt: float = 0.05) -> Dict[str, Any]:
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

