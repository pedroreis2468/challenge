"""
=============================================================================
FSUminho Autonomous Systems & Embedded Software (AS / ESW) Challenge
CONTRATO DE DADOS E ESTRUTURAS DE COMUNICAÇÃO
=============================================================================
Este ficheiro define as estruturas e formatos de dados oficiais utilizados
entre os módulos de Perceção, Planeamento, Controlo, Telemetria e Segurança.
Garante que todas as equipas e funções comunicam com a mesma convenção.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict, Any
import numpy as np

# =============================================================================
# 1. ESTADOS DO SISTEMA AUTÓNOMO (REGRAS FSG DRIVERLESS)
# =============================================================================
class ASState(Enum):
    AS_OFF = "AS_OFF"              # Sistema autónomo desligado
    AS_READY = "AS_READY"          # Sistema inicializado, pronto a arrancar (TS ativo)
    AS_DRIVING = "AS_DRIVING"      # Missão ativa, veículo a navegar autonomamente
    AS_EMERGENCY = "AS_EMERGENCY"  # EBS acionado (travagem de emergência / SDC aberto)

# =============================================================================
# 2. ESTRUTURA DE DADOS DE CONES (PERCEÇÃO)
# =============================================================================
@dataclass
class Cone:
    x: float             # Posição X (metros)
    y: float             # Posição Y (metros)
    color: str           # 'blue' (limite esquerdo), 'yellow' (limite direito), 'orange' (meta/partida), 'unknown'
    confidence: float    # Confiança da deteção [0.0 a 1.0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'x': float(self.x),
            'y': float(self.y),
            'color': str(self.color),
            'confidence': float(self.confidence)
        }

# =============================================================================
# 3. ESTADO DINÂMICO DO VEÍCULO (ODOMETRIA / SLAM)
# =============================================================================
@dataclass
class CarState:
    x: float             # Posição X global (m)
    y: float             # Posição Y global (m)
    yaw: float           # Orientação / Ângulo de rumo (radianos, 0 = eixo +X)
    v: float             # Velocidade linear do veículo (m/s)
    steer: float         # Ângulo de direção atual das rodas dianteiras (radianos)
    yaw_rate: float = 0.0# Velocidade angular em Z (rad/s)

# =============================================================================
# 4. PACOTE DE TELEMETRIA AO VIVO (ESW / CAN BUS)
# =============================================================================
@dataclass
class TelemetryPacket:
    timestamp: float        # Tempo decorrido (s)
    est_car_velocity: float # Velocidade estimada do carro (m/s)
    fl_wheel_speed: float   # Roda Dianteira Esquerda (m/s)
    fr_wheel_speed: float   # Roda Dianteira Direita (m/s)
    rl_wheel_speed: float   # Roda Traseira Esquerda (m/s)
    rr_wheel_speed: float   # Roda Traseira Direita (m/s)
    accel_x: float          # Aceleração longitudinal IMU (m/s^2)
    accel_y: float          # Aceleração lateral IMU (m/s^2)
    gyro_z: float           # Yaw Rate IMU (deg/s)
    apps: float             # Acelerador APPS (0 a 100%)
    brake_pressure: float   # Pressão de travagem (0 a 80 bar)
    as_state: str           # Estado atual: AS_READY, AS_DRIVING, AS_EMERGENCY
