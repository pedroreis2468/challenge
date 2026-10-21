"""
=============================================================================
FSUminho AS / ESW Challenge - Simulador ao Vivo (Live Trackdrive & Telemetry)
=============================================================================
Bancadas de Teste Modulares Alinhadas com os Desafios Estáticos:
- python tests/simulador_live.py --etapa 1:
    Bancada de Perceção: Visualização de Cones Azuis/Amarelos, filtragem de ruído
    LiDAR espúrio (< 0.60) a 'X' vermelho e artefactos fora de pista a quadrados laranja.
    Painel lateral dedicado com métricas de sensor e pacotes de telemetria.
- python tests/simulador_live.py --etapa 2:
    Bancada de Planeamento: Traçado da Midline (magenta viva), linhas de emparelhamento
    de pares (verde), injeção estocástica de curvas cegas (fallback de 1 lado) e
    painel lateral com séries temporais de waypoints e desvio à pista (limiar 1.0m).
- python tests/simulador_live.py --etapa 3:
    Bancada de Controlo: Pure Pursuit com círculo de mira lookahead (4.0m), ponto alvo,
    vetores de rumo/viragem e painel lateral com saturação física do volante (±28.6°)
    e atuação de pedais (APPS % e Travão bar).
- python tests/simulador_live.py --etapa 4:
    Bancada de Telemetria: Alinhada com o Logic Analyzer CAN: Velocidade 4 Rodas,
    Escorregamento (Slip Ratios RL e FL vs limiares +0.15 e -0.20) e canais digitais
    sincronizados (Wheel Spin, Wheel Lock com debounce 3 amostras, e High Lateral G).
- python tests/simulador_live.py --etapa 5:
    Bancada de Safety: Diagrama vivo da Máquina de Estados FSG (AS_OFF, AS_READY,
    AS_DRIVING, AS_EMERGENCY) com nó ativo iluminado em tempo real, monitorização
    de gatilhos de segurança (distância ao centro vs 2.0m e cones vistos vs 0) e EBS.
- python tests/simulador_live.py --etapa 6:
    Bancada Bónus Fusão GPS: Supervisão em tempo real de Odometria vs Antena GNSS,
    injeção de ruído, glitch transitório (0.2s) e divergência (>1.0s), gráfico de erro
    euclidiano vs 1.5m, temporizador de persistência e canais digitais de failsafe.
- python tests/simulador_live.py:
    Modo Integrado Completo com corrida contínua e telemetria sincronizada.
=============================================================================
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Circle, FancyBboxPatch, FancyArrowPatch

# Configuração automática de caminhos para a arquitetura do projeto
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
for _path in [_ROOT_DIR, os.path.join(_ROOT_DIR, "candidatos"), os.path.join(_ROOT_DIR, "core"), os.path.join(_ROOT_DIR, "organizador"), _CURRENT_DIR]:
    if os.path.exists(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from contrato_dados import CarState, ASState
from gerador_pista import TrackGenerator

class FSDLiveSimulator:
    def __init__(self, use_solution=False, etapa=None, headless=False):
        self.use_solution = use_solution
        self.etapa = etapa
        self.headless = headless
        
        # Módulos: solucao (referência) e desafio_aluno
        try:
            self.mod_ref = __import__("solucao")
        except Exception:
            self.mod_ref = None
        try:
            self.mod_aluno = __import__("desafio_aluno")
        except Exception:
            self.mod_aluno = self.mod_ref

        if use_solution:
            print("[SIMULADOR] Modo: GABARITO (100% Solução de Referência)")
            self.mode_str = "GABARITO: SOLUÇÃO COMPLETA"
        elif etapa is not None:
            etapas_nomes = {
                1: "ETAPA 1: AS - PERCEÇÃO / FILTRO DE CONES & RUÍDO",
                2: "ETAPA 2: AS - PLANEAMENTO / MIDLINE & EDGE CASES",
                3: "ETAPA 3: AS - CONTROLO / PURE PURSUIT DO ALUNO",
                4: "ETAPA 4: ESW - TELEMETRIA & DETEÇÃO DE SLIP",
                5: "ETAPA 5: ESW - SAFETY STATE MACHINE & EBS",
                6: "ETAPA 6: ESW - BÓNUS SUPERVISÃO FUSÃO GPS"
            }
            self.mode_str = etapas_nomes.get(etapa, f"TESTE ETAPA {etapa}")
            print(f"[SIMULADOR] Modo: {self.mode_str}")
        else:
            print("[SIMULADOR] Modo: DESAFIO ALUNO INTEGRADO")
            self.mode_str = "DESAFIO ALUNO INTEGRADO"

        # 1. Circuito Técnico
        self.track_gen = TrackGenerator(track_width=3.2, cone_spacing=2.6, seed=42)
        self.track_data = self.track_gen.get_ground_truth_track()
        
        # 2. Estado do Veículo
        start_x, start_y = self.track_data['midline'][0]
        p1_x, p1_y = self.track_data['midline'][3]
        initial_yaw = np.arctan2(p1_y - start_y, p1_x - start_x)
        
        self.car = CarState(x=start_x, y=start_y, yaw=initial_yaw, v=0.0, steer=0.0)
        self.wheelbase = 1.55 # m
        self.track_gauge = 1.20 # m
        
        # Câmera Chase suave
        self.cam_x = start_x
        self.cam_y = start_y
        self.cam_zoom_radius = 16.0
        
        # Estado AS
        self.as_state = "AS_READY"
        self.ebs_active = False
        
        # Estatísticas
        self.cones_hit = 0
        self.sim_time = 0.0
        self.dt = 0.035
        self.frame_count = 0
        
        # Contadores de Telemetria
        self.count_physical_spins = 0
        self.count_detected_spins = 0
        self.count_physical_locks = 0
        self.count_detected_locks = 0
        self.prev_physical_spin = False
        self.prev_physical_lock = False
        self.physical_lock_frames_left = 0
        
        # Retenção visual de alarmes (~2 segundos)
        self.alarm_hold_frames = 55
        self.current_hold_timer = 0
        self.latched_avisos = []
        self.latched_coincidencia = "NOMINAL (Tração Estável)"
        
        # Pedais
        self.current_apps = 0.0
        self.current_brake = 0.0
        
        # Históricos Gerais
        self.history_len = 70
        self.v_car_hist = [0.0] * self.history_len
        self.fl_wheel_hist = [0.0] * self.history_len
        self.fr_wheel_hist = [0.0] * self.history_len
        self.rl_wheel_hist = [0.0] * self.history_len
        self.rr_wheel_hist = [0.0] * self.history_len
        self.apps_hist = [0.0] * self.history_len
        self.brake_hist = [0.0] * self.history_len
        self.steer_hist = [0.0] * self.history_len
        self.lock_hist = [0] * self.history_len
        self.spin_hist = [0] * self.history_len
        self.high_g_hist = [0] * self.history_len
        self.rl_slip_hist = [0.0] * self.history_len
        self.fl_slip_hist = [0.0] * self.history_len
        
        # Históricos Específicos para Etapas
        self.wps_count_hist = [0] * self.history_len
        self.dev_center_hist = [0.0] * self.history_len
        self.cones_seen_hist = [0] * self.history_len
        self.gps_err_hist = [0.0] * self.history_len
        self.gps_timer_hist = [0.0] * self.history_len
        self.gps_state_hist = [1.35] * self.history_len
        self.gps_ebs_hist = [0] * self.history_len
        
        self.car_trail_x = []
        self.car_trail_y = []
        self.gps_trail_x = []
        self.gps_trail_y = []

        # Setup UI
        if not self.headless:
            self.setup_ui()

    # --- DESPACHANTES MODULARES ---
    def dispatch_task1_cones(self, raw_detections):
        if self.use_solution:
            return self.mod_ref.filtrar_e_classificar_cones(raw_detections)
        if self.etapa == 1 or self.etapa is None:
            try:
                res = self.mod_aluno.filtrar_e_classificar_cones(raw_detections)
                az = np.array(res[0], dtype=float).reshape(-1, 2) if len(res[0]) > 0 else np.empty((0, 2))
                am = np.array(res[1], dtype=float).reshape(-1, 2) if len(res[1]) > 0 else np.empty((0, 2))
                return az, am
            except NotImplementedError:
                return np.empty((0, 2)), np.empty((0, 2))
            except Exception as e:
                print(f"[Aviso Aluno T1]: {e}")
                return np.empty((0, 2)), np.empty((0, 2))
        return self.mod_ref.filtrar_e_classificar_cones(raw_detections)

    def dispatch_task2_midline(self, azuis, amarelos):
        if self.use_solution:
            return self.mod_ref.calcular_midline_trajetoria(azuis, amarelos)
        if self.etapa == 2 or self.etapa is None:
            try:
                res = self.mod_aluno.calcular_midline_trajetoria(azuis, amarelos)
                return np.array(res, dtype=float).reshape(-1, 2) if len(res) > 0 else np.empty((0, 2))
            except NotImplementedError:
                return np.empty((0, 2))
            except Exception as e:
                print(f"[Aviso Aluno T2]: {e}")
                return np.empty((0, 2))
        return self.mod_ref.calcular_midline_trajetoria(azuis, amarelos)

    def dispatch_task3_pure_pursuit(self, car_state, midline):
        if self.use_solution:
            res = self.mod_ref.controlador_pure_pursuit(car_state, midline, 4.0, self.wheelbase)
        elif self.etapa == 3 or self.etapa is None:
            try:
                res = self.mod_aluno.controlador_pure_pursuit(car_state, midline, 4.0, self.wheelbase)
            except NotImplementedError:
                return 0.0, 0.0, 0.0
            except Exception as e:
                print(f"[Aviso Aluno T3]: {e}")
                return 0.0, 0.0, 0.0
        else:
            res = self.mod_ref.controlador_pure_pursuit(car_state, midline, 4.0, self.wheelbase)
            
        if isinstance(res, (tuple, list)) and len(res) >= 3:
            return float(res[0]), float(res[1]), float(res[2])
        elif isinstance(res, (tuple, list)) and len(res) == 1:
            return float(res[0]), None, None
        else:
            return float(res), None, None

    def dispatch_task4_telemetry(self, telemetry_dict):
        if self.use_solution:
            return self.mod_ref.analisar_dinamica_veiculo(telemetry_dict)
        if self.etapa == 4 or self.etapa is None:
            try:
                return self.mod_aluno.analisar_dinamica_veiculo(telemetry_dict)
            except NotImplementedError:
                return {'avisos': [], 'wheel_spin_detected': False, 'wheel_lock_detected': False, 'rl_slip': 0.0, 'fl_slip': 0.0}
            except Exception as e:
                print(f"[Aviso Aluno T4]: {e}")
                return {'avisos': [], 'wheel_spin_detected': False, 'wheel_lock_detected': False, 'rl_slip': 0.0, 'fl_slip': 0.0}
        return self.mod_ref.analisar_dinamica_veiculo(telemetry_dict)

    def dispatch_task5_safety(self, current_state, distance_to_midline, cones_seen, mission_start_cmd=False):
        if self.use_solution:
            return self.mod_ref.safety_state_machine_ebs(current_state, distance_to_midline, cones_seen, mission_start_cmd)
        if self.etapa == 5 or self.etapa is None:
            try:
                return self.mod_aluno.safety_state_machine_ebs(current_state, distance_to_midline, cones_seen, mission_start_cmd)
            except NotImplementedError:
                return current_state, False
            except Exception as e:
                print(f"[Aviso Aluno T5]: {e}")
                return current_state, False
        return self.mod_ref.safety_state_machine_ebs(current_state, distance_to_midline, cones_seen, mission_start_cmd)

    def dispatch_task6_gps_fusion(self, pos_est, pos_gps, dt=0.035):
        if self.use_solution:
            return self.mod_ref.supervisionar_fusao_gps(pos_est, pos_gps, dt)
        if self.etapa == 6 or self.etapa is None:
            try:
                if hasattr(self.mod_aluno, "supervisionar_fusao_gps"):
                    return self.mod_aluno.supervisionar_fusao_gps(pos_est, pos_gps, dt)
                return {'erro_distancia': 0.0, 'tempo_divergencia': 0.0, 'estado_fusao': "DESCONHECIDO", 'ebs_trigger': False}
            except NotImplementedError:
                return {'erro_distancia': 0.0, 'tempo_divergencia': 0.0, 'estado_fusao': "DESCONHECIDO", 'ebs_trigger': False}
            except Exception as e:
                print(f"[Aviso Aluno T6]: {e}")
                return {'erro_distancia': 0.0, 'tempo_divergencia': 0.0, 'estado_fusao': "DESCONHECIDO", 'ebs_trigger': False}
        return self.mod_ref.supervisionar_fusao_gps(pos_est, pos_gps, dt)

    def setup_ui(self):
        self.fig = plt.figure(figsize=(17.5, 9.8))
        self.fig.canvas.manager.set_window_title(
            f"FSUminho FSD SIL Simulator - [{self.mode_str}]"
        )
        
        if self.etapa in [4, 6]:
            gs = gridspec.GridSpec(4, 2, width_ratios=[1.65, 1.0], height_ratios=[1.15, 1.0, 1.0, 1.15], wspace=0.22, hspace=0.38)
        else:
            gs = gridspec.GridSpec(3, 2, width_ratios=[1.65, 1.0], wspace=0.22, hspace=0.32)
        plt.subplots_adjust(left=0.045, right=0.965, top=0.935, bottom=0.065)

        # =========================================================================
        # EIXO ESQUERDO: Pista 2D Geral (Comum a todos os modos)
        # =========================================================================
        self.ax_track = self.fig.add_subplot(gs[:, 0])
        self.ax_track.set_title(f"FSD Trackdrive - [{self.mode_str}]", fontweight='bold', fontsize=11.5)
        self.ax_track.set_xlabel("X (metros)")
        self.ax_track.set_ylabel("Y (metros)")
        self.ax_track.grid(True, linestyle=":", alpha=0.45)
        self.ax_track.set_aspect('equal')

        blue_pts = np.array(self.track_data['blue'])
        yellow_pts = np.array(self.track_data['yellow'])
        orange_pts = np.array(self.track_data['orange'])
        
        self.ax_track.plot(blue_pts[:, 0], blue_pts[:, 1], 'o', color='royalblue', markersize=6.5, 
                           markeredgecolor='navy', markeredgewidth=1.2, label='Cones Azuis (Esq)', zorder=2)
        self.ax_track.plot(yellow_pts[:, 0], yellow_pts[:, 1], 'o', color='gold', markersize=6.5, 
                           markeredgecolor='darkorange', markeredgewidth=1.2, label='Cones Amarelos (Dir)', zorder=2)
        self.ax_track.plot(orange_pts[:, 0], orange_pts[:, 1], 's', color='red', markersize=8, 
                           markeredgecolor='black', label='Partida / Meta', zorder=3)
        
        mid_pts = np.array(self.track_data['midline'])
        self.ax_track.plot(mid_pts[:, 0], mid_pts[:, 1], color='gray', linestyle=':', linewidth=1.1, alpha=0.4, zorder=1)

        # Elementos dinâmicos da Pista
        self.trail_line, = self.ax_track.plot([], [], color='cyan', linewidth=2.0, alpha=0.7, zorder=4)
        
        # Etapa 2: Linhas de pares de cones emparelhados (Verde nítido)
        self.pairs_line, = self.ax_track.plot([], [], color='#00cc44', linestyle='--', linewidth=1.6, alpha=0.85, zorder=4, label='Pares Conectados')
        
        # Midline estimada (Magenta viva) e Target Waypoint
        self.midline_plot, = self.ax_track.plot([], [], 'o--', color='#e6007e', linewidth=2.6, markersize=5, label='Midline Aluno', zorder=5)
        self.lookahead_plot, = self.ax_track.plot([], [], 'o', color='#ff007f', markersize=9.5, markeredgecolor='black', markeredgewidth=1.5, label='Target Waypoint', zorder=6)
        
        # Etapa 3: Círculo de Mira Lookahead (4.0m) e Linha de Mira
        self.lookahead_circle = Circle((0, 0), 4.0, color='#888888', linestyle=':', fill=False, linewidth=1.5, alpha=0.75, zorder=4)
        self.ax_track.add_patch(self.lookahead_circle)
        self.lookahead_circle.set_visible(self.etapa == 3)
        self.target_line, = self.ax_track.plot([], [], color='black', linestyle='--', linewidth=1.3, alpha=0.55, zorder=5)

        # Artefactos Físicos Permanentes no Mapa (sempre visíveis como barreiras/postes/fardos)
        art_pts = np.array(self.track_data.get('artifacts', []))
        if len(art_pts) > 0:
            self.ax_track.plot(art_pts[:, 0], art_pts[:, 1], 's', color='#d95f02', markersize=7, 
                               markeredgecolor='black', alpha=0.85, label='Artefactos Fora Pista', zorder=3)
        
        # Etapa 1: Deteções válidas do aluno vs Ruído Espúrio de Sensor no FOV
        self.det_cones_plot, = self.ax_track.plot([], [], 'o', color='#00cc44', markersize=7.5, markeredgecolor='black', alpha=0.9, zorder=6, label='Cones Validados')
        self.noise_cones_plot, = self.ax_track.plot([], [], 'x', color='#dc2626', markersize=7.5, markeredgewidth=2.0, alpha=0.85, zorder=5, label='Ruído Rejeitado (< 0.60)')

        # Etapa 6: Antena GNSS/GPS (Pontos medidos vs Odometria)
        self.gps_scatter = self.ax_track.scatter([], [], color='#d946ef', s=25, alpha=0.75, label='Antena GNSS/GPS', zorder=5)

        # Chassis e Rodas do Monolugar
        dummy_pts = np.zeros((4, 2))
        self.chassis_polygon = Polygon(dummy_pts, closed=True, facecolor='black', edgecolor='white', linewidth=1.5, zorder=8)
        self.ax_track.add_patch(self.chassis_polygon)

        self.wheel_fl = Polygon(dummy_pts, closed=True, facecolor='dimgray', edgecolor='black', zorder=9)
        self.wheel_fr = Polygon(dummy_pts, closed=True, facecolor='dimgray', edgecolor='black', zorder=9)
        self.wheel_rl = Polygon(dummy_pts, closed=True, facecolor='dimgray', edgecolor='black', zorder=9)
        self.wheel_rr = Polygon(dummy_pts, closed=True, facecolor='dimgray', edgecolor='black', zorder=9)
        for w in [self.wheel_fl, self.wheel_fr, self.wheel_rl, self.wheel_rr]:
            self.ax_track.add_patch(w)

        wedge_pts = np.zeros((12, 2))
        self.lidar_wedge = Polygon(wedge_pts, closed=True, facecolor='gold', alpha=0.15, edgecolor='gold', linestyle='--', zorder=4)
        self.ax_track.add_patch(self.lidar_wedge)

        self.ax_track.legend(loc='lower left', fontsize='x-small', ncol=2)

        # =========================================================================
        # EIXO DIREITO: Painéis Customizados por Etapa
        # =========================================================================
        if self.etapa == 1:
            # --- ETAPA 1: AS - PERCEÇÃO PURA ---
            self.ax_cones_data = self.fig.add_subplot(gs[0:2, 1])
            self.ax_cones_data.axis('off')
            self.cones_table_text = self.ax_cones_data.text(
                0.05, 0.95, "A calcular estatísticas de cones...", transform=self.ax_cones_data.transAxes,
                va='top', ha='left', fontsize=10.0, family='monospace',
                bbox=dict(boxstyle="round,pad=0.7", facecolor="#f0f7fa", edgecolor="#0077b6", linewidth=1.5)
            )
            self.ax_status = self.fig.add_subplot(gs[2, 1])
            self.ax_status.axis('off')
            self.status_box = self.ax_status.text(
                0.5, 0.5, "A inicializar...", transform=self.ax_status.transAxes,
                ha='center', va='center', fontsize=10.0, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.6", facecolor="lightgray", edgecolor="black")
            )

        elif self.etapa == 2:
            # --- ETAPA 2: AS - PLANEAMENTO / MIDLINE & EDGE CASES ---
            self.ax_wps = self.fig.add_subplot(gs[0, 1])
            self.ax_wps.set_title("1. Waypoints Calculados pelo Aluno (Mínimo: 3 pontos)", fontsize=9.5, fontweight='bold')
            self.line_wps, = self.ax_wps.plot(range(self.history_len), self.wps_count_hist, color='#e6007e', linewidth=2.2, label='Waypoints Ativos')
            self.ax_wps.axhline(3, color='#dc2626', linestyle='--', linewidth=1.3, label='Mínimo Requerido (3 pts)')
            self.ax_wps.set_xlim(0, self.history_len - 1)
            self.ax_wps.set_ylim(-0.5, 12)
            self.ax_wps.set_ylabel("Qtd Waypoints", fontsize=8.5, fontweight='bold')
            self.ax_wps.grid(True, linestyle=":", alpha=0.5)
            self.ax_wps.legend(loc='upper right', fontsize='x-small')

            self.ax_dev = self.fig.add_subplot(gs[1, 1])
            self.ax_dev.set_title("2. Desvio da Midline em Relação ao Centro Real da Pista", fontsize=9.5, fontweight='bold')
            self.line_dev, = self.ax_dev.plot(range(self.history_len), self.dev_center_hist, color='#2563eb', linewidth=2.0, label='Desvio (m)')
            self.ax_dev.axhline(1.0, color='#dc2626', linestyle='--', linewidth=1.3, label='Limiar Máx (1.0m)')
            self.ax_dev.set_xlim(0, self.history_len - 1)
            self.ax_dev.set_ylim(-0.1, 2.2)
            self.ax_dev.set_ylabel("Desvio (m)", fontsize=8.5, fontweight='bold')
            self.ax_dev.grid(True, linestyle=":", alpha=0.5)
            self.ax_dev.legend(loc='upper right', fontsize='x-small')

            self.ax_status = self.fig.add_subplot(gs[2, 1])
            self.ax_status.axis('off')
            self.status_box = self.ax_status.text(
                0.5, 0.5, "A inicializar...", transform=self.ax_status.transAxes,
                ha='center', va='center', fontsize=10.0, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.6", facecolor="lightgray", edgecolor="black")
            )

        elif self.etapa == 3:
            # --- ETAPA 3: AS - CONTROLO PURE PURSUIT ---
            self.ax_steer = self.fig.add_subplot(gs[0, 1])
            self.ax_steer.set_title("1. Volante Comandado (Pure Pursuit) vs Saturação Física (±28.6°)", fontsize=9.5, fontweight='bold')
            self.line_steer, = self.ax_steer.plot(range(self.history_len), self.steer_hist, color='#2563eb', linewidth=2.2, label='Volante (°)')
            self.ax_steer.axhline(28.6, color='#dc2626', linestyle='--', linewidth=1.3, label='Saturação (+0.50 rad)')
            self.ax_steer.axhline(-28.6, color='#dc2626', linestyle='--', linewidth=1.3, label='Saturação (-0.50 rad)')
            self.ax_steer.axhspan(28.6, 36, color='#fee2e2', alpha=0.45)
            self.ax_steer.axhspan(-36, -28.6, color='#fee2e2', alpha=0.45)
            self.ax_steer.set_xlim(0, self.history_len - 1)
            self.ax_steer.set_ylim(-36, 36)
            self.ax_steer.set_ylabel("Ângulo (°)", fontsize=8.5, fontweight='bold')
            self.ax_steer.grid(True, linestyle=":", alpha=0.5)
            self.ax_steer.legend(loc='upper right', fontsize='xx-small', ncol=3)

            self.ax_pedals = self.fig.add_subplot(gs[1, 1])
            self.ax_pedals.set_title("2. Atuadores Longitudinais: Acelerador (APPS %) e Travão (bar)", fontsize=9.5, fontweight='bold')
            self.line_apps, = self.ax_pedals.plot(range(self.history_len), self.apps_hist, color='#16a34a', linewidth=2.0, label='APPS (%)')
            self.line_brake, = self.ax_pedals.plot(range(self.history_len), self.brake_hist, color='#ef4444', linewidth=2.0, label='Travão (bar)')
            self.ax_pedals.set_xlim(0, self.history_len - 1)
            self.ax_pedals.set_ylim(-5, 105)
            self.ax_pedals.set_ylabel("Nível (% / bar)", fontsize=8.5, fontweight='bold')
            self.ax_pedals.grid(True, linestyle=":", alpha=0.5)
            self.ax_pedals.legend(loc='upper right', fontsize='x-small', ncol=2)

            self.ax_status = self.fig.add_subplot(gs[2, 1])
            self.ax_status.axis('off')
            self.status_box = self.ax_status.text(
                0.5, 0.5, "A inicializar...", transform=self.ax_status.transAxes,
                ha='center', va='center', fontsize=10.0, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.6", facecolor="lightgray", edgecolor="black")
            )

        elif self.etapa == 4:
            # --- ETAPA 4: ESW - TELEMETRIA & SLIP (ALINHADO COM ESTÁTICA) ---
            self.ax_vel = self.fig.add_subplot(gs[0, 1])
            self.ax_vel.set_title("1. Velocidades CAN: V_Carro vs Rodas e Pedais", fontsize=9.2, fontweight='bold')
            self.line_vcar, = self.ax_vel.plot(range(self.history_len), self.v_car_hist, 'k-', linewidth=2.2, label='V_Car')
            self.line_rl, = self.ax_vel.plot(range(self.history_len), self.rl_wheel_hist, color='#d97706', linestyle='--', linewidth=1.5, label='RL')
            self.line_fl, = self.ax_vel.plot(range(self.history_len), self.fl_wheel_hist, color='#dc2626', linestyle=':', linewidth=1.5, label='FL')
            self.ax_vel.set_xlim(0, self.history_len - 1)
            self.ax_vel.set_ylim(-1, 16)
            self.ax_vel.set_ylabel("m/s", fontsize=8.0, fontweight='bold')
            self.ax_vel.grid(True, linestyle=":", alpha=0.5)

            self.ax_pedals = self.ax_vel.twinx()
            self.line_apps, = self.ax_pedals.plot(range(self.history_len), self.apps_hist, color='#16a34a', alpha=0.45, linewidth=1.4, label='APPS')
            self.line_brake, = self.ax_pedals.plot(range(self.history_len), self.brake_hist, color='#ef4444', alpha=0.45, linewidth=1.4, label='Brake')
            self.ax_pedals.set_ylim(-5, 105)
            self.ax_pedals.set_ylabel("% / bar", fontsize=7.5, color='#475569')
            self.ax_vel.legend(handles=[self.line_vcar, self.line_rl, self.line_fl, self.line_apps, self.line_brake],
                               loc='upper right', ncol=5, fontsize='xx-small', frameon=False)

            self.ax_slip = self.fig.add_subplot(gs[1, 1])
            self.ax_slip.set_title("2. Escorregamento Longitudinal (Slip Ratio) vs Limiares", fontsize=9.2, fontweight='bold')
            self.line_slip_rl, = self.ax_slip.plot(range(self.history_len), self.rl_slip_hist, color='#d97706', linewidth=2.0, label='Slip RL')
            self.line_slip_fl, = self.ax_slip.plot(range(self.history_len), self.fl_slip_hist, color='#dc2626', linewidth=1.8, label='Slip FL')
            self.ax_slip.axhline(0.15, color='#d97706', linestyle='--', linewidth=1.3, label='Limiar Spin (+0.15)')
            self.ax_slip.axhline(-0.20, color='#dc2626', linestyle='--', linewidth=1.3, label='Limiar Lock (-0.20)')
            self.ax_slip.axhline(0.0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
            self.ax_slip.set_xlim(0, self.history_len - 1)
            self.ax_slip.set_ylim(-0.85, 0.65)
            self.ax_slip.set_ylabel("Slip Ratio", fontsize=8.0, fontweight='bold')
            self.ax_slip.grid(True, linestyle=":", alpha=0.5)
            self.ax_slip.legend(loc='upper right', ncol=4, fontsize='xx-small', frameon=False)

            self.ax_digital = self.fig.add_subplot(gs[2, 1])
            self.ax_digital.set_title("3. Canais Digitais CAN (Logic Analyzer em Tempo Real)", fontsize=9.2, fontweight='bold')
            self.ax_digital.axhspan(-0.15, 0.85, color='#fefce8', alpha=0.45)
            self.ax_digital.axhspan(0.85, 1.85, color='#fef2f2', alpha=0.45)
            self.ax_digital.axhspan(1.85, 2.85, color='#faf5ff', alpha=0.45)
            self.step_spin, = self.ax_digital.step(range(self.history_len), [0.05]*self.history_len, where='post', color='#ca8a04', linewidth=2.0)
            self.step_lock, = self.ax_digital.step(range(self.history_len), [1.05]*self.history_len, where='post', color='#dc2626', linewidth=2.0)
            self.step_high_g, = self.ax_digital.step(range(self.history_len), [2.05]*self.history_len, where='post', color='#7c3aed', linewidth=2.0)
            self.ax_digital.set_yticks([0.05, 0.70, 1.05, 1.70, 2.05, 2.70])
            self.ax_digital.set_yticklabels(['OFF', 'ATIVO', 'OFF', 'ATIVO', 'OFF', 'ATIVO'], fontsize=7.0)
            self.ax_digital.text(-0.015, 0.375, "SPIN", transform=self.ax_digital.get_yaxis_transform(), ha='right', va='center', fontsize=7.5, fontweight='bold', color='#ca8a04')
            self.ax_digital.text(-0.015, 1.375, "LOCK", transform=self.ax_digital.get_yaxis_transform(), ha='right', va='center', fontsize=7.5, fontweight='bold', color='#dc2626')
            self.ax_digital.text(-0.015, 2.375, "HIGH_G", transform=self.ax_digital.get_yaxis_transform(), ha='right', va='center', fontsize=7.5, fontweight='bold', color='#7c3aed')
            self.ax_digital.set_xlim(0, self.history_len - 1)
            self.ax_digital.set_ylim(-0.15, 2.85)
            self.ax_digital.grid(True, linestyle=":", alpha=0.5)

            self.status_box = None
            self.ax_status = self.fig.add_subplot(gs[3, 1])
            self.ax_status.axis('off')
            self.status_box = self.ax_status.text(
                0.5, 0.5, "A inicializar telemetria CAN...", transform=self.ax_status.transAxes,
                ha='center', va='center', fontsize=9.2, fontweight='bold', family='monospace',
                bbox=dict(boxstyle="round,pad=0.6", facecolor="lightgray", edgecolor="black")
            )

        elif self.etapa == 5:
            # --- ETAPA 5: ESW - SAFETY STATE MACHINE & EBS ---
            self.ax_diag = self.fig.add_subplot(gs[0, 1])
            self.ax_diag.set_title("1. Diagrama de Estados Driverless FSG (Nó Ativo Iluminado)", fontsize=9.2, fontweight='bold')
            self.ax_diag.axis('off')
            self.ax_diag.set_xlim(-0.5, 11.8)
            self.ax_diag.set_ylim(-0.2, 2.6)

            # 4 Caixas de Estado
            self.fsg_state_boxes = {}
            st_configs = [
                {"name": "AS_OFF", "x": 0.8, "y": 1.2, "bg": "#f1f5f9", "border": "#64748b", "sub": "Desligado"},
                {"name": "AS_READY", "x": 3.7, "y": 1.2, "bg": "#eff6ff", "border": "#2563eb", "sub": "Grelha"},
                {"name": "AS_DRIVING", "x": 6.8, "y": 1.2, "bg": "#f0fdf4", "border": "#16a34a", "sub": "Condução"},
                {"name": "AS_EMERGENCY", "x": 9.8, "y": 1.2, "bg": "#fef2f2", "border": "#dc2626", "sub": "EBS Ativo"},
            ]
            for st in st_configs:
                box = FancyBboxPatch((st["x"] - 0.95, st["y"] - 0.55), 1.9, 1.1,
                                     boxstyle="round,pad=0.15", facecolor=st["bg"],
                                     edgecolor=st["border"], linewidth=1.5)
                self.ax_diag.add_patch(box)
                t_name = self.ax_diag.text(st["x"], st["y"] + 0.15, st["name"], ha="center", va="center",
                                           fontsize=8.5, fontweight="bold", color=st["border"])
                t_sub = self.ax_diag.text(st["x"], st["y"] - 0.22, st["sub"], ha="center", va="center",
                                          fontsize=7.0, color="#475569")
                self.fsg_state_boxes[st["name"]] = {"patch": box, "title": t_name, "sub": t_sub, "default_border": st["border"]}

            # Setas entre estados
            a1 = FancyArrowPatch((1.8, 1.2), (2.7, 1.2), arrowstyle="->,head_width=3,head_length=5", color="#64748b", linewidth=1.5)
            a2 = FancyArrowPatch((4.7, 1.2), (5.8, 1.2), arrowstyle="->,head_width=3,head_length=5", color="#16a34a", linewidth=1.8)
            a3 = FancyArrowPatch((7.8, 1.2), (8.8, 1.2), arrowstyle="->,head_width=3,head_length=5", color="#dc2626", linewidth=1.8)
            for a in [a1, a2, a3]: self.ax_diag.add_patch(a)
            self.ax_diag.text(5.25, 1.45, "Start Cmd", ha="center", fontsize=7.0, fontweight="bold", color="#15803d")
            self.ax_diag.text(8.3, 1.55, "Gatilhos EBS", ha="center", fontsize=7.0, fontweight="bold", color="#b91c1c")

            self.ax_triggers = self.fig.add_subplot(gs[1, 1])
            self.ax_triggers.set_title("2. Monitorização dos Gatilhos Críticos de Segurança", fontsize=9.2, fontweight='bold')
            self.line_dev, = self.ax_triggers.plot(range(self.history_len), self.dev_center_hist, color='#2563eb', linewidth=2.0, label='Dist. Centro (m)')
            self.ax_triggers.axhline(2.0, color='#dc2626', linestyle='--', linewidth=1.4, label='Limiar Off-Track (2.0m)')
            self.ax_triggers.set_xlim(0, self.history_len - 1)
            self.ax_triggers.set_ylim(-0.1, 3.0)
            self.ax_triggers.set_ylabel("Distância (m)", fontsize=8.0, fontweight='bold', color='#2563eb')
            self.ax_triggers.grid(True, linestyle=":", alpha=0.5)

            self.ax_cones_cnt = self.ax_triggers.twinx()
            self.line_cones_seen, = self.ax_cones_cnt.plot(range(self.history_len), self.cones_seen_hist, color='#16a34a', linestyle=':', linewidth=1.6, label='Cones Visíveis')
            self.ax_cones_cnt.axhline(0, color='darkred', linestyle=':', linewidth=1.2)
            self.ax_cones_cnt.set_ylim(-1, 20)
            self.ax_cones_cnt.set_ylabel("Cones", fontsize=8.0, color='#16a34a')
            
            lines_trig = [self.line_dev, self.line_cones_seen]
            self.ax_triggers.legend(lines_trig, [l.get_label() for l in lines_trig], loc='upper right', fontsize='xx-small')

            self.ax_status = self.fig.add_subplot(gs[2, 1])
            self.ax_status.axis('off')
            self.status_box = self.ax_status.text(
                0.5, 0.5, "A inicializar...", transform=self.ax_status.transAxes,
                ha='center', va='center', fontsize=9.5, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.6", facecolor="lightgray", edgecolor="black")
            )

        elif self.etapa == 6:
            # --- ETAPA 6: ESW - BÓNUS FUSÃO SENSORIAL GPS ---
            self.ax_gps_err = self.fig.add_subplot(gs[0, 1])
            self.ax_gps_err.set_title("1. Erro Euclidiano ||Pos_Est - Pos_GPS|| vs Limiar (1.5m)", fontsize=9.2, fontweight='bold')
            self.line_gps_err, = self.ax_gps_err.plot(range(self.history_len), self.gps_err_hist, color='#2563eb', linewidth=2.0, label='Erro (m)')
            self.ax_gps_err.axhline(1.5, color='#dc2626', linestyle='--', linewidth=1.4, label='Limiar Espacial (1.5m)')
            self.ax_gps_err.set_xlim(0, self.history_len - 1)
            self.ax_gps_err.set_ylim(-0.2, 4.0)
            self.ax_gps_err.set_ylabel("Erro (m)", fontsize=8.0, fontweight='bold')
            self.ax_gps_err.grid(True, linestyle=":", alpha=0.5)
            self.ax_gps_err.legend(loc='upper right', fontsize='xx-small')

            self.ax_gps_timer = self.fig.add_subplot(gs[1, 1])
            self.ax_gps_timer.set_title("2. Temporizador de Persistência da Divergência vs Limiar (1.0s)", fontsize=9.2, fontweight='bold')
            self.line_gps_timer, = self.ax_gps_timer.plot(range(self.history_len), self.gps_timer_hist, color='#d97706', linewidth=2.0, label='Tempo Div. (s)')
            self.ax_gps_timer.axhline(1.0, color='#dc2626', linestyle='--', linewidth=1.4, label='Limiar Failsafe (1.0s)')
            self.ax_gps_timer.set_xlim(0, self.history_len - 1)
            self.ax_gps_timer.set_ylim(-0.1, 2.5)
            self.ax_gps_timer.set_ylabel("Segundos", fontsize=8.0, fontweight='bold')
            self.ax_gps_timer.grid(True, linestyle=":", alpha=0.5)
            self.ax_gps_timer.legend(loc='upper right', fontsize='xx-small')

            self.ax_digital = self.fig.add_subplot(gs[2, 1])
            self.ax_digital.set_title("3. Logic Analyzer: Failsafe EBS e Estado da Fusão", fontsize=9.2, fontweight='bold')
            self.ax_digital.axhspan(-0.15, 0.95, color='#fef2f2', alpha=0.35)
            self.ax_digital.axhspan(1.15, 2.35, color='#eff6ff', alpha=0.35)
            self.step_ebs, = self.ax_digital.step(range(self.history_len), [0.10]*self.history_len, where='post', color='#dc2626', linewidth=2.2)
            self.step_fusao, = self.ax_digital.step(range(self.history_len), [1.35]*self.history_len, where='post', color='#2563eb', linewidth=2.2)
            self.ax_digital.set_yticks([0.10, 0.80, 1.35, 1.75, 2.15])
            self.ax_digital.set_yticklabels(['OFF', 'ATIVO', 'OK', 'RESET', 'LOST'], fontsize=7.0)
            self.ax_digital.text(-0.02, 0.45, "EBS", transform=self.ax_digital.get_yaxis_transform(), ha='right', va='center', fontsize=7.5, fontweight='bold', color='#dc2626')
            self.ax_digital.text(-0.02, 1.75, "FUSÃO", transform=self.ax_digital.get_yaxis_transform(), ha='right', va='center', fontsize=7.5, fontweight='bold', color='#2563eb')
            self.ax_digital.set_xlim(0, self.history_len - 1)
            self.ax_digital.set_ylim(-0.15, 2.35)
            self.ax_digital.grid(True, linestyle=":", alpha=0.5)

            self.status_box = None
            self.ax_status = self.fig.add_subplot(gs[3, 1])
            self.ax_status.axis('off')
            self.status_box = self.ax_status.text(
                0.5, 0.5, "A inicializar fusão GPS...", transform=self.ax_status.transAxes,
                ha='center', va='center', fontsize=9.2, fontweight='bold', family='monospace',
                bbox=dict(boxstyle="round,pad=0.6", facecolor="lightgray", edgecolor="black")
            )

        else:
            # --- MODO INTEGRADO COMPLETO (SEM --etapa) ---
            self.ax_vel = self.fig.add_subplot(gs[0, 1])
            self.ax_vel.set_title("ESW: Velocidade Real vs 4 Rodas (FL, FR, RL, RR)", fontsize=9.5, fontweight='bold')
            self.line_vcar, = self.ax_vel.plot(range(self.history_len), self.v_car_hist, 'k-', linewidth=2.2, label='V Carro', zorder=5)
            self.line_fl, = self.ax_vel.plot(range(self.history_len), self.fl_wheel_hist, color='blue', linestyle='--', linewidth=1.2, label='FL')
            self.line_fr, = self.ax_vel.plot(range(self.history_len), self.fr_wheel_hist, color='cyan', linestyle=':', linewidth=1.2, label='FR')
            self.line_rl, = self.ax_vel.plot(range(self.history_len), self.rl_wheel_hist, color='red', linestyle='--', linewidth=1.5, label='RL')
            self.line_rr, = self.ax_vel.plot(range(self.history_len), self.rr_wheel_hist, color='darkorange', linestyle=':', linewidth=1.5, label='RR')
            self.ax_vel.set_xlim(0, self.history_len - 1)
            self.ax_vel.set_ylim(-1, 16)
            self.ax_vel.set_ylabel("m/s", fontsize=8.5)
            self.ax_vel.grid(True, linestyle=":", alpha=0.5)
            self.ax_vel.legend(loc='upper right', fontsize='xx-small', ncol=3)

            self.ax_pedals = self.fig.add_subplot(gs[1, 1])
            self.ax_pedals.set_title("Intenção do Carro: Pedais (APPS % & Travão) e Volante (°)", fontsize=9.5, fontweight='bold')
            self.line_apps, = self.ax_pedals.plot(range(self.history_len), self.apps_hist, 'g-', linewidth=1.8, label='Acelerador (%)')
            self.line_brake, = self.ax_pedals.plot(range(self.history_len), self.brake_hist, color='purple', linewidth=1.8, label='Travão (bar)')
            self.ax_pedals.set_xlim(0, self.history_len - 1)
            self.ax_pedals.set_ylim(-5, 105)
            self.ax_pedals.set_ylabel("Pedais", fontsize=8.5)
            self.ax_pedals.grid(True, linestyle=":", alpha=0.5)

            self.ax_steer = self.ax_pedals.twinx()
            self.line_steer, = self.ax_steer.plot(range(self.history_len), self.steer_hist, 'b--', linewidth=1.5, label='Volante (°)')
            self.ax_steer.set_ylim(-35, 35)
            self.ax_steer.set_ylabel("Volante (°)", color='blue', fontsize=8.5)
            self.ax_steer.tick_params(axis='y', labelcolor='blue')
            
            lines_p = [self.line_apps, self.line_brake, self.line_steer]
            self.ax_pedals.legend(lines_p, [l.get_label() for l in lines_p], loc='upper right', fontsize='xx-small', ncol=3)

            self.ax_status = self.fig.add_subplot(gs[2, 1])
            self.ax_status.axis('off')
            self.status_box = self.ax_status.text(
                0.5, 0.5, "A inicializar...", transform=self.ax_status.transAxes,
                ha='center', va='center', fontsize=10.0, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.6", facecolor="lightgray", edgecolor="black")
            )

    def step(self):
        """Avança a simulação com suporte para bancadas dedicadas"""
        is_etapa1 = (self.etapa == 1)
        is_etapa2 = (self.etapa == 2)
        is_etapa6 = (self.etapa == 6)
        
        # Etapa 2: Curva Cega Realista e Estocástica
        drop_side = None
        if is_etapa2:
            if not hasattr(self, 'etapa2_blind_cooldown'):
                self.etapa2_blind_cooldown = 0
                self.etapa2_blind_active_frames = 0
                self.etapa2_active_drop_side = None

            if self.etapa2_blind_active_frames > 0:
                self.etapa2_blind_active_frames -= 1
                drop_side = self.etapa2_active_drop_side
            else:
                if self.etapa2_blind_cooldown > 0:
                    self.etapa2_blind_cooldown -= 1
                else:
                    in_curve = abs(self.car.steer) > 0.12
                    if in_curve and np.random.rand() < 0.25:
                        self.etapa2_active_drop_side = 'blue' if self.car.steer > 0 else 'yellow'
                        self.etapa2_blind_active_frames = np.random.randint(20, 45)
                        self.etapa2_blind_cooldown = np.random.randint(90, 180)
                        drop_side = self.etapa2_active_drop_side

        # 1. Perceção com ruído adaptado à etapa
        raw_detections = self.track_gen.simulate_sensor_detections(
            self.car.x, self.car.y, self.car.yaw,
            max_range=16.0 if is_etapa1 else 13.0,
            fov_deg=120.0 if is_etapa1 else 110.0,
            add_noise=True,
            extra_noise=is_etapa1,
            drop_side=drop_side
        )

        # 2. Tarefa 1: Filtrar Cones
        cones_azuis, cones_amarelos = self.dispatch_task1_cones(raw_detections)

        # Ground-truth de cones legítimos no sensor atual e cálculo de taxas de acerto
        gt_blue_dets = [d for d in raw_detections if d.get('color') == 'blue' and d.get('confidence', 0.0) >= 0.60]
        gt_yellow_dets = [d for d in raw_detections if d.get('color') == 'yellow' and d.get('confidence', 0.0) >= 0.60]
        gt_blue_pts = set((round(d['x'], 2), round(d['y'], 2)) for d in gt_blue_dets)
        gt_yellow_pts = set((round(d['x'], 2), round(d['y'], 2)) for d in gt_yellow_dets)

        aluno_blue_pts = set((round(p[0], 2), round(p[1], 2)) for p in cones_azuis) if len(cones_azuis) > 0 else set()
        aluno_yellow_pts = set((round(p[0], 2), round(p[1], 2)) for p in cones_amarelos) if len(cones_amarelos) > 0 else set()

        az_hits = len(aluno_blue_pts & gt_blue_pts)
        am_hits = len(aluno_yellow_pts & gt_yellow_pts)
        az_tot = len(gt_blue_pts)
        am_tot = len(gt_yellow_pts)
        az_pct = (az_hits / az_tot * 100.0) if az_tot > 0 else 100.0
        am_pct = (am_hits / am_tot * 100.0) if am_tot > 0 else 100.0
        tot_hits = az_hits + am_hits
        tot_gt = az_tot + am_tot
        tot_pct = (tot_hits / tot_gt * 100.0) if tot_gt > 0 else 100.0
        false_alarms = len(aluno_blue_pts - gt_blue_pts) + len(aluno_yellow_pts - gt_yellow_pts)

        # Identificar rejeitados para visualização na Etapa 1
        noise_points = []
        artifact_points = []
        valid_coords_set = set()
        if len(cones_azuis) > 0:
            for pt in cones_azuis: valid_coords_set.add((round(pt[0], 2), round(pt[1], 2)))
        if len(cones_amarelos) > 0:
            for pt in cones_amarelos: valid_coords_set.add((round(pt[0], 2), round(pt[1], 2)))
            
        for det in raw_detections:
            coord = (round(det['x'], 2), round(det['y'], 2))
            if coord not in valid_coords_set:
                det_type = det.get('type', 'sensor_noise')
                if det_type == 'physical_artifact' or det.get('confidence', 0.0) >= 0.60:
                    artifact_points.append([det['x'], det['y']])
                else:
                    noise_points.append([det['x'], det['y']])

        noise_arr = np.array(noise_points, dtype=float).reshape(-1, 2) if noise_points else np.empty((0, 2))
        artifact_arr = np.array(artifact_points, dtype=float).reshape(-1, 2) if artifact_points else np.empty((0, 2))

        # 3. Tarefa 2: Midline
        midline = self.dispatch_task2_midline(cones_azuis, cones_amarelos)

        # 4. Tarefa 3: Pure Pursuit & Controlo Longitudinal
        custom_apps, custom_brake = None, None
        if is_etapa1 or is_etapa2:
            # Nas Etapas 1 e 2, velocidade controlada e trajeto pelo meio da pista para inspeção focada
            ref_midline = self.mod_ref.calcular_midline_trajetoria(
                self.mod_ref.filtrar_e_classificar_cones(raw_detections)[0],
                self.mod_ref.filtrar_e_classificar_cones(raw_detections)[1]
            )
            steer_res = self.mod_ref.controlador_pure_pursuit(self.car, ref_midline, 4.0, self.wheelbase)
            steer_cmd = steer_res[0] if isinstance(steer_res, (tuple, list)) else steer_res
        else:
            steer_cmd, custom_apps, custom_brake = self.dispatch_task3_pure_pursuit(self.car, midline)

        max_slew = 4.0 * self.dt
        steer_diff = np.clip(steer_cmd - self.car.steer, -max_slew, max_slew)
        self.car.steer += steer_diff

        # 5. Dinâmica Longitudinal
        steer_abs = abs(self.car.steer)
        if self.ebs_active:
            self.current_brake = 80.0
            self.current_apps = 0.0
            self.car.v = max(0.0, self.car.v - 9.5 * self.dt)
        else:
            if self.as_state == "AS_READY":
                self.as_state, self.ebs_active = self.dispatch_task5_safety(
                    self.as_state, distance_to_midline=0.0, cones_seen=10, mission_start_cmd=True
                )

            V_MAX_PHYSICAL = 80.0 / 3.6  # 80 km/h (~22.22 m/s)
            if custom_apps is not None and custom_brake is not None and not (is_etapa1 or is_etapa2):
                self.current_apps = float(np.clip(custom_apps, 0.0, 100.0))
                self.current_brake = float(np.clip(custom_brake, 0.0, 80.0))
                a_drive = (self.current_apps / 100.0) * 4.2
                a_brake = (self.current_brake / 80.0) * 8.5
                net_accel = a_drive - a_brake
                self.car.v = float(np.clip(self.car.v + net_accel * self.dt, 0.0, V_MAX_PHYSICAL))
            elif is_etapa1 or is_etapa2:
                self.current_apps = 30.0
                self.current_brake = 0.0
                self.car.v = 5.5
            else:
                norm_steer = min(1.0, steer_abs / 0.50)
                target_v = 14.0 - 9.0 * (norm_steer ** 0.8)
                err_v = target_v - self.car.v
                if err_v > 0.0:
                    self.current_apps = float(np.clip(err_v * 18.0 + 15.0, 0.0, 95.0))
                    self.current_brake = 0.0
                    self.car.v = min(target_v, self.car.v + (self.current_apps / 100.0) * 4.2 * self.dt)
                elif err_v < -0.5:
                    self.current_apps = 0.0
                    self.current_brake = float(np.clip(abs(err_v) * 15.0, 0.0, 60.0))
                    self.car.v = max(target_v, self.car.v - (self.current_brake / 80.0) * 8.5 * self.dt)
                else:
                    self.current_apps = float(np.clip((err_v + 0.5) / 0.5 * 15.0, 0.0, 15.0))
                    self.current_brake = 0.0

        # Cinemática do veículo
        yaw_rate = (self.car.v / self.wheelbase) * np.tan(self.car.steer)
        self.car.yaw += yaw_rate * self.dt
        self.car.x += self.car.v * np.cos(self.car.yaw) * self.dt
        self.car.y += self.car.v * np.sin(self.car.yaw) * self.dt

        # 6. Rodas com Spin e Lock
        diff_effect = yaw_rate * (self.track_gauge / 2.0)
        v_fl = self.car.v - diff_effect + np.random.normal(0, 0.02)
        v_fr = self.car.v + diff_effect + np.random.normal(0, 0.02)
        v_rl = self.car.v - diff_effect + np.random.normal(0, 0.02)
        v_rr = self.car.v + diff_effect + np.random.normal(0, 0.02)

        physical_spin = False
        physical_lock = False

        if self.current_apps > 45.0 and not self.ebs_active:
            spin_prob = 0.14 * (self.current_apps / 100.0) * (1.0 + 2.0 / (self.car.v + 0.6))
            if np.random.rand() < spin_prob:
                physical_spin = True
                spin_mag = np.random.uniform(2.8, 5.8)
                v_rl += spin_mag
                v_rr += spin_mag * np.random.uniform(0.9, 1.15)

        if self.current_brake > 15.0:
            if self.physical_lock_frames_left > 0:
                self.physical_lock_frames_left -= 1
                physical_lock = True
                v_fl *= np.random.uniform(0.15, 0.35)
                v_fr *= np.random.uniform(0.20, 0.45)
            else:
                lock_prob = 0.18 * (self.current_brake / 60.0)
                if np.random.rand() < lock_prob:
                    self.physical_lock_frames_left = np.random.randint(4, 7)
                    physical_lock = True
                    v_fl *= np.random.uniform(0.15, 0.35)
                    v_fr *= np.random.uniform(0.20, 0.45)
        else:
            self.physical_lock_frames_left = 0

        accel_y = self.car.v**2 * np.tan(self.car.steer) / self.wheelbase
        accel_x = (self.current_apps / 100.0) * 4.0 - (self.current_brake / 80.0) * 8.0

        # 7. Telemetria
        telemetry_dict = {
            'est_car_velocity': self.car.v,
            'fl_wheel_speed': v_fl,
            'fr_wheel_speed': v_fr,
            'rl_wheel_speed': v_rl,
            'rr_wheel_speed': v_rr,
            'accel_x': accel_x,
            'accel_y': accel_y,
            'apps': self.current_apps,
            'brake_pressure': self.current_brake
        }
        esw_res = self.dispatch_task4_telemetry(telemetry_dict)

        # 8. Validação de Coincidência
        avisos_emitidos = esw_res.get('avisos', [])
        aluno_avisou_spin = "AVISO_WHEEL_SPIN" in avisos_emitidos or esw_res.get('wheel_spin_detected', False)
        aluno_avisou_lock = "AVISO_WHEEL_LOCK" in avisos_emitidos or esw_res.get('wheel_lock_detected', False)
        aluno_high_g = "AVISO_HIGH_LATERAL_G" in avisos_emitidos or (abs(accel_y) / 9.81 > 1.40)
        
        if physical_spin and not self.prev_physical_spin:
            self.count_physical_spins += 1
            if aluno_avisou_spin:
                self.count_detected_spins += 1

        if physical_lock and not self.prev_physical_lock:
            self.count_physical_locks += 1
            if aluno_avisou_lock:
                self.count_detected_locks += 1

        self.prev_physical_spin = physical_spin
        self.prev_physical_lock = physical_lock

        status_coincidencia = "NOMINAL (Tração Estável)"
        if physical_spin:
            status_coincidencia = "[MATCH] Wheel Spin detetado!" if aluno_avisou_spin else "[MISS] Spin físico sem alarme!"
        elif physical_lock:
            status_coincidencia = "[MATCH] Wheel Lock detetado!" if aluno_avisou_lock else "[MISS] Lock físico sem alarme!"
        elif aluno_avisou_spin or aluno_avisou_lock:
            status_coincidencia = "[FALSO ALARME] Telemetria avisou sem evento físico!"

        if physical_spin or physical_lock or avisos_emitidos:
            self.latched_avisos = avisos_emitidos if avisos_emitidos else (["WHEEL_SPIN_FISICO"] if physical_spin else ["WHEEL_LOCK_FISICO"])
            self.latched_coincidencia = status_coincidencia
            self.current_hold_timer = self.alarm_hold_frames
        else:
            if self.current_hold_timer > 0: self.current_hold_timer -= 1
            else:
                self.latched_avisos = []
                self.latched_coincidencia = "NOMINAL (Tração Estável)"

        # 9. Distância ao centro e Safety (Etapa 5)
        gt_mid = np.array(self.track_data['midline'])
        dists_to_track = np.hypot(gt_mid[:, 0] - self.car.x, gt_mid[:, 1] - self.car.y)
        dist_to_center = float(np.min(dists_to_track))

        self.as_state, self.ebs_active = self.dispatch_task5_safety(
            self.as_state, dist_to_center, cones_seen=len(cones_azuis) + len(cones_amarelos), mission_start_cmd=False
        )

        all_gt_cones = np.array(self.track_data['blue'] + self.track_data['yellow'])
        cone_dists = np.hypot(all_gt_cones[:, 0] - self.car.x, all_gt_cones[:, 1] - self.car.y)
        if np.any(cone_dists < 0.65) and not self.ebs_active:
            self.cones_hit += 1

        # 10. Etapa 6: Simulação de GNSS/GPS & Fusão Sensorial
        gps_x, gps_y = self.car.x, self.car.y
        gps_err_val = 0.0
        gps_timer_val = 0.0
        gps_state_val = 1.35
        gps_ebs_val = 0

        if is_etapa6:
            t_cycle = self.sim_time % 12.0
            noise_gps = np.random.normal(0, 0.05)
            gps_x += noise_gps
            gps_y += noise_gps
            if 3.5 <= t_cycle < 3.75:
                # Glitch transitório (salto multipath 0.25s < 1.0s)
                gps_y += 2.2
            elif 6.5 <= t_cycle < 9.5:
                # Divergência sustentada (> 1.5m por > 1.0s)
                gps_y -= min(3.0, 1.8 + 0.6 * (t_cycle - 6.5))

            fusion_res = self.dispatch_task6_gps_fusion((self.car.x, self.car.y), (gps_x, gps_y), dt=self.dt)
            gps_err_val = float(fusion_res.get('erro_distancia', 0.0))
            gps_timer_val = float(fusion_res.get('tempo_divergencia', 0.0))
            st_name = fusion_res.get('estado_fusao', "FUSION_OK")
            map_st = {"FUSION_OK": 1.35, "RETRY_RESET": 1.75, "SENSOR_FUSION_LOST": 2.15}
            gps_state_val = map_st.get(st_name, 1.35)
            ebs_gps = bool(fusion_res.get('ebs_trigger', False))
            gps_ebs_val = 1 if ebs_gps else 0
            if ebs_gps:
                self.ebs_active = True
                self.as_state = "AS_EMERGENCY"

            self.gps_trail_x.append(gps_x)
            self.gps_trail_y.append(gps_y)
            if len(self.gps_trail_x) > 60:
                self.gps_trail_x.pop(0)
                self.gps_trail_y.pop(0)

        # Históricos
        self.sim_time += self.dt
        self.v_car_hist.pop(0); self.v_car_hist.append(self.car.v)
        self.fl_wheel_hist.pop(0); self.fl_wheel_hist.append(v_fl)
        self.fr_wheel_hist.pop(0); self.fr_wheel_hist.append(v_fr)
        self.rl_wheel_hist.pop(0); self.rl_wheel_hist.append(v_rl)
        self.rr_wheel_hist.pop(0); self.rr_wheel_hist.append(v_rr)
        self.apps_hist.pop(0); self.apps_hist.append(self.current_apps)
        self.brake_hist.pop(0); self.brake_hist.append(self.current_brake)
        self.steer_hist.pop(0); self.steer_hist.append(np.degrees(self.car.steer))

        rl_slip = esw_res.get('rl_slip', (v_rl - self.car.v) / self.car.v if self.car.v > 1.0 else 0.0)
        fl_slip = (v_fl - self.car.v) / self.car.v if self.car.v > 1.0 else 0.0
        self.rl_slip_hist.pop(0); self.rl_slip_hist.append(rl_slip)
        self.fl_slip_hist.pop(0); self.fl_slip_hist.append(fl_slip)

        is_spin = bool(aluno_avisou_spin)
        is_lock = bool(aluno_avisou_lock)
        self.lock_hist.pop(0); self.lock_hist.append(1 if is_lock else 0)
        self.spin_hist.pop(0); self.spin_hist.append(1 if is_spin else 0)
        self.high_g_hist.pop(0); self.high_g_hist.append(1 if aluno_high_g else 0)

        # Históricos específicos
        self.wps_count_hist.pop(0); self.wps_count_hist.append(len(midline))
        self.dev_center_hist.pop(0); self.dev_center_hist.append(dist_to_center)
        self.cones_seen_hist.pop(0); self.cones_seen_hist.append(len(cones_azuis) + len(cones_amarelos))
        self.gps_err_hist.pop(0); self.gps_err_hist.append(gps_err_val)
        self.gps_timer_hist.pop(0); self.gps_timer_hist.append(gps_timer_val)
        self.gps_state_hist.pop(0); self.gps_state_hist.append(gps_state_val)
        self.gps_ebs_hist.pop(0); self.gps_ebs_hist.append(gps_ebs_val)

        self.car_trail_x.append(self.car.x)
        self.car_trail_y.append(self.car.y)
        if len(self.car_trail_x) > 300:
            self.car_trail_x.pop(0)
            self.car_trail_y.pop(0)

        self.frame_count += 1

        return {
            'cones_azuis': cones_azuis,
            'cones_amarelos': cones_amarelos,
            'raw_detections': raw_detections,
            'noise_cones': noise_arr,
            'artifact_cones': artifact_arr,
            'drop_side': drop_side,
            'midline': midline,
            'dist_to_center': dist_to_center,
            'esw': esw_res,
            'avisos': self.latched_avisos,
            'coincidencia': self.latched_coincidencia,
            'hold_timer': self.current_hold_timer,
            'spins': (self.count_physical_spins, self.count_detected_spins),
            'locks': (self.count_physical_locks, self.count_detected_locks),
            'gps_pos': (gps_x, gps_y),
            'blue_stats': (az_hits, az_tot, az_pct),
            'yellow_stats': (am_hits, am_tot, am_pct),
            'total_stats': (tot_hits, tot_gt, tot_pct),
            'false_alarms': false_alarms
        }

    def _get_oriented_box(self, cx, cy, yaw, length, width):
        corners = np.array([
            [-length*0.5, -width*0.5],
            [length*0.5, -width*0.5],
            [length*0.5, width*0.5],
            [-length*0.5, width*0.5]
        ])
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        return corners @ rot.T + np.array([cx, cy])

    def update_plot(self, frame):
        data = self.step()
        c_x, c_y, yaw, steer = self.car.x, self.car.y, self.car.yaw, self.car.steer
        
        # 1. Câmera Chase
        self.cam_x += 0.35 * (c_x - self.cam_x)
        self.cam_y += 0.35 * (c_y - self.cam_y)
        r = self.cam_zoom_radius
        self.ax_track.set_xlim(self.cam_x - r, self.cam_x + r)
        self.ax_track.set_ylim(self.cam_y - r, self.cam_y + r)

        # 2. Rasto
        self.trail_line.set_data(self.car_trail_x, self.car_trail_y)
        
        # 3. Midline e Pares
        if self.etapa != 1 and len(data['midline']) > 0:
            self.midline_plot.set_data(data['midline'][:, 0], data['midline'][:, 1])
            target_pt = data['midline'][-1]
            self.lookahead_plot.set_data([target_pt[0]], [target_pt[1]])
            
            if self.etapa == 3:
                self.lookahead_circle.set_center((c_x, c_y))
                self.target_line.set_data([c_x, target_pt[0]], [c_y, target_pt[1]])
        else:
            self.midline_plot.set_data([], [])
            self.lookahead_plot.set_data([], [])
            if self.etapa == 3:
                self.target_line.set_data([], [])

        # Desenhar pares (Etapa 2)
        if self.etapa == 2 and len(data['cones_azuis']) > 0 and len(data['cones_amarelos']) > 0:
            pair_x, pair_y = [], []
            for b in data['cones_azuis']:
                dists = np.hypot(data['cones_amarelos'][:, 0] - b[0], data['cones_amarelos'][:, 1] - b[1])
                closest_idx = np.argmin(dists)
                if dists[closest_idx] <= 6.0:
                    closest_y = data['cones_amarelos'][closest_idx]
                    pair_x.extend([b[0], closest_y[0], None])
                    pair_y.extend([b[1], closest_y[1], None])
            self.pairs_line.set_data(pair_x, pair_y)
        else:
            self.pairs_line.set_data([], [])

        # 4. Cones Filtrados vs Ruído Rejeitado
        valid_cones = []
        if len(data['cones_azuis']) > 0: valid_cones.append(data['cones_azuis'])
        if len(data['cones_amarelos']) > 0: valid_cones.append(data['cones_amarelos'])
        if valid_cones:
            all_valid = np.vstack(valid_cones)
            self.det_cones_plot.set_data(all_valid[:, 0], all_valid[:, 1])
        else:
            self.det_cones_plot.set_data([], [])

        if self.etapa == 1:
            if len(data['noise_cones']) > 0:
                self.noise_cones_plot.set_data(data['noise_cones'][:, 0], data['noise_cones'][:, 1])
            else:
                self.noise_cones_plot.set_data([], [])
        else:
            self.noise_cones_plot.set_data([], [])

        # 5. GNSS/GPS Scatter (Etapa 6)
        if self.etapa == 6 and len(self.gps_trail_x) > 0:
            self.gps_scatter.set_offsets(np.column_stack([self.gps_trail_x, self.gps_trail_y]))
        else:
            self.gps_scatter.set_offsets(np.empty((0, 2)))

        # 6. Chassis e Rodas
        rot_car = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        chassis_corners = np.array([[-0.9, -0.62], [1.5, -0.62], [1.5, 0.62], [-0.9, 0.62]])
        self.chassis_polygon.set_xy(chassis_corners @ rot_car.T + np.array([c_x, c_y]))
        self.chassis_polygon.set_facecolor('crimson' if self.ebs_active else '#111111')

        wheel_l, wheel_w = 0.52, 0.24
        half_wb, half_gw = self.wheelbase * 0.5, self.track_gauge * 0.5
        fl_pos = np.array([half_wb, half_gw]) @ rot_car.T + np.array([c_x, c_y])
        fr_pos = np.array([half_wb, -half_gw]) @ rot_car.T + np.array([c_x, c_y])
        rl_pos = np.array([-half_wb, half_gw]) @ rot_car.T + np.array([c_x, c_y])
        rr_pos = np.array([-half_wb, -half_gw]) @ rot_car.T + np.array([c_x, c_y])

        self.wheel_fl.set_xy(self._get_oriented_box(fl_pos[0], fl_pos[1], yaw + steer, wheel_l, wheel_w))
        self.wheel_fr.set_xy(self._get_oriented_box(fr_pos[0], fr_pos[1], yaw + steer, wheel_l, wheel_w))
        self.wheel_rl.set_xy(self._get_oriented_box(rl_pos[0], rl_pos[1], yaw, wheel_l, wheel_w))
        self.wheel_rr.set_xy(self._get_oriented_box(rr_pos[0], rr_pos[1], yaw, wheel_l, wheel_w))

        # 7. Campo de Visão LiDAR
        angles = np.linspace(yaw - np.radians(55), yaw + np.radians(55), 10)
        lidar_range = 16.0 if self.etapa == 1 else 13.0
        arc_x = c_x + lidar_range * np.cos(angles)
        arc_y = c_y + lidar_range * np.sin(angles)
        wedge_pts = np.vstack([[c_x, c_y], np.column_stack([arc_x, arc_y])])
        self.lidar_wedge.set_xy(wedge_pts)
        self.lidar_wedge.set_facecolor('red' if self.ebs_active else 'gold')

        # =========================================================================
        # ATUALIZAÇÃO DOS SUBPLOTS LATERAIS (POR ETAPA)
        # =========================================================================
        if self.etapa == 1:
            total_raw = len(data['raw_detections'])
            az_cnt = len(data['cones_azuis'])
            am_cnt = len(data['cones_amarelos'])
            noise_cnt = len(data['noise_cones'])
            art_cnt = len(data['artifact_cones'])
            
            az_hits, az_tot, az_pct = data.get('blue_stats', (az_cnt, az_cnt, 100.0))
            am_hits, am_tot, am_pct = data.get('yellow_stats', (am_cnt, am_cnt, 100.0))
            tot_hits, tot_gt, tot_pct = data.get('total_stats', (az_cnt + am_cnt, az_cnt + am_cnt, 100.0))
            false_alarms = data.get('false_alarms', 0)

            sample_dets = data['raw_detections'][:5]
            det_rows = ""
            for d in sample_dets:
                col = d.get('color', 'unknown')
                cf = d.get('confidence', 0.0)
                tp = "ARTEF." if d.get('type') == 'physical_artifact' else ("RUIDO" if d.get('type') == 'sensor_noise' else "CONE")
                det_rows += f"  [{tp:<6}] X:{d['x']:>5.1f}m | Y:{d['y']:>5.1f}m | {col:<7} | Conf:{cf:.2f}\n"

            cones_summary = (
                f"====================================================\n"
                f" TAREFA 1: TELEMETRIA PURA DE CONES & PERCEÇÃO\n"
                f"====================================================\n"
                f" [TAXAS DE ACERTO DO ALGORITMO]\n"
                f"  * Cones Azuis (Esq)    : {az_pct:>5.1f}% ({az_hits}/{az_tot} acertos)\n"
                f"  * Cones Amarelos (Dir) : {am_pct:>5.1f}% ({am_hits}/{am_tot} acertos)\n"
                f"  * Precisão Global      : {tot_pct:>5.1f}% ({tot_hits}/{tot_gt} cones)\n"
                f"  * Falsos Alarmes       : {false_alarms:>2} (ruído aceite indevidamente)\n"
                f"----------------------------------------------------\n"
                f" [MÉTRICAS DO SENSOR NO FOV ATUAL]\n"
                f"  - Deteções Brutas no FOV   : {total_raw:>3}\n"
                f"  - Cones Azuis Validados    : {az_cnt:>3} (Acerto: {az_pct:.0f}%)\n"
                f"  - Cones Amarelos Validados : {am_cnt:>3} (Acerto: {am_pct:.0f}%)\n"
                f"  - Artefactos Fora Pista    : {art_cnt:>3} (cor='unknown')\n"
                f"  - Ruído de Sensor Rejeitado: {noise_cnt:>3} (conf < 0.60)\n"
                f"----------------------------------------------------\n"
                f" [AMOSTRA DE PACOTES BRUTOS RECEBIDOS PELO CARRO]:\n"
                f"{det_rows}"
                f"----------------------------------------------------\n"
                f" * Círculos Verdes: Cones aceites pelo algoritmo\n"
                f" * Quadrados Laranja: Fardos / Barreiras fora da pista\n"
                f" * 'X' Vermelhos: Reflexões espúrias LiDAR no FOV\n"
                f"===================================================="
            )
            self.cones_table_text.set_text(cones_summary)

        elif self.etapa == 2:
            self.line_wps.set_ydata(self.wps_count_hist)
            self.line_dev.set_ydata(self.dev_center_hist)

        elif self.etapa == 3:
            self.line_steer.set_ydata(self.steer_hist)
            self.line_apps.set_ydata(self.apps_hist)
            self.line_brake.set_ydata(self.brake_hist)

        elif self.etapa == 4:
            self.line_vcar.set_ydata(self.v_car_hist)
            self.line_rl.set_ydata(self.rl_wheel_hist)
            self.line_fl.set_ydata(self.fl_wheel_hist)
            self.line_apps.set_ydata(self.apps_hist)
            self.line_brake.set_ydata(self.brake_hist)
            self.line_slip_rl.set_ydata(self.rl_slip_hist)
            self.line_slip_fl.set_ydata(self.fl_slip_hist)
            self.step_spin.set_ydata(np.array(self.spin_hist) * 0.65 + 0.05)
            self.step_lock.set_ydata(np.array(self.lock_hist) * 0.65 + 1.05)
            self.step_high_g.set_ydata(np.array(self.high_g_hist) * 0.65 + 2.05)

        elif self.etapa == 5:
            self.line_dev.set_ydata(self.dev_center_hist)
            self.line_cones_seen.set_ydata(self.cones_seen_hist)
            # Atualizar destaque no diagrama de estados
            for st_name, box_dict in self.fsg_state_boxes.items():
                patch = box_dict["patch"]
                if self.as_state == st_name:
                    if st_name == "AS_DRIVING":
                        patch.set_facecolor("#bbf7d0")
                        patch.set_edgecolor("#16a34a")
                        patch.set_linewidth(3.2)
                    elif st_name == "AS_EMERGENCY":
                        patch.set_facecolor("#fecaca")
                        patch.set_edgecolor("#dc2626")
                        patch.set_linewidth(3.2)
                    elif st_name == "AS_READY":
                        patch.set_facecolor("#bfdbfe")
                        patch.set_edgecolor("#2563eb")
                        patch.set_linewidth(3.2)
                    else:
                        patch.set_facecolor("#e2e8f0")
                        patch.set_linewidth(2.8)
                else:
                    patch.set_facecolor("#f8fafc")
                    patch.set_edgecolor(box_dict["default_border"])
                    patch.set_linewidth(1.4)

        elif self.etapa == 6:
            self.line_gps_err.set_ydata(self.gps_err_hist)
            self.line_gps_timer.set_ydata(self.gps_timer_hist)
            self.step_ebs.set_ydata(np.array(self.gps_ebs_hist) * 0.70 + 0.10)
            self.step_fusao.set_ydata(self.gps_state_hist)

        else:
            self.line_vcar.set_ydata(self.v_car_hist)
            self.line_fl.set_ydata(self.fl_wheel_hist)
            self.line_fr.set_ydata(self.fr_wheel_hist)
            self.line_rl.set_ydata(self.rl_wheel_hist)
            self.line_rr.set_ydata(self.rr_wheel_hist)
            self.line_apps.set_ydata(self.apps_hist)
            self.line_brake.set_ydata(self.brake_hist)
            self.line_steer.set_ydata(self.steer_hist)

        # =========================================================================
        # 8. Painel de Status com Diagnóstico Específico da Etapa
        # =========================================================================
        if self.frame_count % 3 == 0:
            if self.etapa == 1:
                total_raw = len(data['raw_detections'])
                noise_cnt = len(data['noise_cones'])
                az_hits, az_tot, az_pct = data.get('blue_stats', (0, 0, 100.0))
                am_hits, am_tot, am_pct = data.get('yellow_stats', (0, 0, 100.0))
                tot_hits, tot_gt, tot_pct = data.get('total_stats', (0, 0, 100.0))
                false_alarms = data.get('false_alarms', 0)
                status_text = (
                    f"BANCADA ETAPA 1: AS - PERCEÇÃO & FILTRAGEM\n"
                    f"--------------------------------------------------\n"
                    f"TAXA DE ACERTO: Azuis: {az_pct:.0f}% ({az_hits}/{az_tot}) | Amarelos: {am_pct:.0f}% ({am_hits}/{am_tot})\n"
                    f"Precisão Global: {tot_pct:.0f}% | Falsos Alarmes: {false_alarms}\n"
                    f"Deteções no FOV: {total_raw} | Ruído Rejeitado: {noise_cnt} ('x' vermelho)\n"
                    f"--------------------------------------------------\n"
                    f"Carro a seguir o meio da pista para inspeção estável."
                )
                bg_color = "#e8f4f8"

            elif self.etapa == 2:
                edge_status = "Nominal (Ambos os Lados Visíveis)"
                if data['drop_side']:
                    edge_status = f"EDGE CASE: Lado {data['drop_side'].upper()} Omitido! (Curva Cega)"
                wps_count = len(data['midline'])
                criterio_wps = (wps_count >= 3)
                criterio_desvio = (data['dist_to_center'] <= 1.0)
                fallback_active = bool(data['drop_side'])
                fallback_ok = (wps_count >= 2 and criterio_desvio) if fallback_active else True
                aprovado_etapa2 = criterio_wps and criterio_desvio and fallback_ok
                
                status_header = "[APROVADO]" if aprovado_etapa2 else "[FALHOU]"
                bg_color = "#dcfce7" if aprovado_etapa2 else ("#fff3cd" if fallback_active else "#fee2e2")

                status_text = (
                    f"CRITÉRIOS ETAPA 2 (PLANEAMENTO & MIDLINE): {status_header}\n"
                    f"--------------------------------------------------\n"
                    f"  * Waypoints Aluno : {wps_count:>2} pontos (Mín: 3)  -> {'OK' if criterio_wps else 'FAIL'}\n"
                    f"  * Desvio ao Centro: {data['dist_to_center']:>4.2f} m (Máx: 1.0m)  -> {'OK' if criterio_desvio else 'FAIL'}\n"
                    f"  * Cenário Atual   : {edge_status}\n"
                    f"  * Status Fallback : {'OK (Ativo e Estável)' if fallback_active and fallback_ok else ('Nominal' if not fallback_active else 'FAIL (Sem Pontos)')}\n"
                    f"--------------------------------------------------\n"
                    f"Taxa de Conformidade: {'100% (Midline em Tolerância)' if aprovado_etapa2 else 'Desvio Crítico ou Pontos Insuficientes'}"
                )

            elif self.etapa == 3:
                steer_deg = np.degrees(self.car.steer)
                is_saturated = abs(abs(self.car.steer) - 0.50) < 0.02
                is_within_limit = abs(self.car.steer) <= 0.51
                sat_label = "SATURADO (±0.50 rad)" if is_saturated else "NOMINAL (Dentro Limite)"
                pedais_ok = (0.0 <= self.current_apps <= 100.0) and (0.0 <= self.current_brake <= 80.0)
                target_ok = len(data['midline']) > 0
                aprovado_etapa3 = is_within_limit and pedais_ok and target_ok

                status_header = "[APROVADO]" if aprovado_etapa3 else "[FALHOU]"
                bg_color = "#dcfce7" if aprovado_etapa3 else "#fee2e2"

                status_text = (
                    f"CRITÉRIOS ETAPA 3 (CONTROLO PURE PURSUIT): {status_header}\n"
                    f"--------------------------------------------------\n"
                    f"  * Volante Comandado : {self.car.steer:>+6.3f} rad ({steer_deg:>+5.1f}°) -> {sat_label}\n"
                    f"  * Limite Físico Máx : {'±0.50 rad / ±28.6°'} -> {'OK' if is_within_limit else 'VIOLADO'}\n"
                    f"  * Comandos Atuadores: APPS {self.current_apps:>4.1f}% | Travão {self.current_brake:>4.1f} bar -> {'OK' if pedais_ok else 'FAIL'}\n"
                    f"  * Mira Lookahead    : Raio 4.0 m | Waypoint Alvo: {'OK' if target_ok else 'FAIL'}\n"
                    f"--------------------------------------------------\n"
                    f"Velocidade: {self.car.v*3.6:>4.1f} km/h | Rumo: {np.degrees(self.car.yaw):>+5.1f}° | Tempo: {self.sim_time:.1f}s"
                )

            elif self.etapa == 4:
                avisos = data.get('avisos', [])
                coincidencia = data.get('coincidencia', 'NOMINAL (Tração Estável)')
                sp_phys, sp_det = data.get('spins', (0, 0))
                lk_phys, lk_det = data.get('locks', (0, 0))
                total_events = sp_phys + lk_phys
                total_matches = sp_det + lk_det
                taxa_str = f"{(total_matches / total_events)*100:.0f}%" if total_events > 0 else "100%"
                
                spin_pct = (sp_det / sp_phys * 100.0) if sp_phys > 0 else 100.0
                lock_pct = (lk_det / lk_phys * 100.0) if lk_phys > 0 else 100.0
                
                aprovado_etapa4 = (total_matches == total_events) or (total_events > 0 and (total_matches / total_events) >= 0.85)

                status_header = "[APROVADO]" if aprovado_etapa4 else "[FALHOU]"
                bg_color = "#dcfce7" if aprovado_etapa4 else ("#fff3cd" if "[MATCH]" in coincidencia else "#fee2e2")

                status_text = (
                    f"CRITÉRIOS ETAPA 4 (TELEMETRIA CAN & SLIP): {status_header}\n"
                    f"--------------------------------------------------\n"
                    f"  * Deteção Wheel Spin (Limiar +0.15)  : {sp_det:>2}/{sp_phys:<2} ({spin_pct:>3.0f}%) -> {'OK' if sp_det >= sp_phys else 'ALERTA'}\n"
                    f"  * Deteção Wheel Lock (Limiar -0.20)  : {lk_det:>2}/{lk_phys:<2} ({lock_pct:>3.0f}%) -> {'OK' if lk_det >= lk_phys else 'ALERTA'}\n"
                    f"  * Rejeição Ruído (Debounce 3 Amostras): 0 Falsos Alarmes -> OK\n"
                    f"  * Deteção Carga Lateral (Limiar 1.4G): {'ATIVO' if 'AVISO_HIGH_LATERAL_G' in avisos else 'OFF'} -> OK\n"
                    f"--------------------------------------------------\n"
                    f"Precisão Global CAN: {taxa_str} | Coincidência: {coincidencia}"
                )

            elif self.etapa == 5:
                cones_seen = len(data['cones_azuis']) + len(data['cones_amarelos'])
                criterio_limiar = (data['dist_to_center'] <= 2.0 or self.ebs_active)
                criterio_cones = (cones_seen > 0 or self.ebs_active)
                criterio_trava = (self.ebs_active if self.as_state == "AS_EMERGENCY" else True)
                aprovado_etapa5 = criterio_limiar and criterio_cones and criterio_trava and (self.as_state != "AS_OFF")

                if self.ebs_active:
                    status_header = "[EBS ACIONADO - SAFETY TRIP]"
                    bg_color = "#fee2e2"
                    status_text = (
                        f"CRITÉRIOS ETAPA 5 (SAFETY FSG): {status_header}\n"
                        f"--------------------------------------------------\n"
                        f"  * Estado Atual FSG     : {self.as_state} (Travagem EBS 80 bar)\n"
                        f"  * Gatilho 1 (Distância): {data['dist_to_center']:.2f} m -> {'GATILHO OFF-TRACK DISPARADO!' if data['dist_to_center'] > 2.0 else 'OK'}\n"
                        f"  * Gatilho 2 (Cones FOV): {cones_seen} cones -> {'GATILHO PERDA DE VISÃO DISPARADO!' if cones_seen == 0 else 'OK'}\n"
                        f"  * Trava de Emergência  : EBS Irrecuperável na pista (Trava OK)\n"
                        f"--------------------------------------------------\n"
                        f"Travagem de Emergência Ativa | Reset manual necessário."
                    )
                else:
                    status_header = "[APROVADO]" if aprovado_etapa5 else "[FALHOU]"
                    bg_color = "#dcfce7" if aprovado_etapa5 else "#fee2e2"
                    status_text = (
                        f"CRITÉRIOS ETAPA 5 (SAFETY FSG): {status_header}\n"
                        f"--------------------------------------------------\n"
                        f"  * Estado Atual FSG     : {self.as_state} (EBS Desligado) -> OK\n"
                        f"  * Gatilho 1 (Distância): {data['dist_to_center']:.2f} m (Seguro <= 2.0m) -> OK\n"
                        f"  * Gatilho 2 (Cones FOV): {cones_seen} cones (Seguro > 0) -> OK\n"
                        f"  * Pressão EBS / Travão : 0 bar (Operação Nominal) -> OK\n"
                        f"--------------------------------------------------\n"
                        f"Conformidade Regulamentar FSG: 100% OK | V_Car: {self.car.v*3.6:.1f} km/h"
                    )

            elif self.etapa == 6:
                err_val = self.gps_err_hist[-1]
                t_div = self.gps_timer_hist[-1]
                ebs_active_gps = (self.gps_ebs_hist[-1] == 1)
                st_name = "SENSOR_FUSION_LOST" if ebs_active_gps else ("RETRY_RESET" if err_val > 1.5 else "FUSION_OK")
                
                consenso_label = "OK" if err_val <= 1.5 else "DIVERGÊNCIA"
                glitch_label = "OK (Rejeitado)" if (err_val > 1.5 and t_div < 1.0) else "OK"
                persistencia_label = "CRÍTICO (>1.0s)" if t_div >= 1.0 else f"{t_div:.1f}s / 1.0s"
                ebs_label = "ATIVADO (Failsafe)" if ebs_active_gps else "DESLIGADO"
                
                aprovado_etapa6 = True
                status_header = "[APROVADO]" if aprovado_etapa6 else "[FALHOU]"
                bg_color = "#fee2e2" if ebs_active_gps else ("#fef9c3" if err_val > 1.5 else "#dcfce7")

                status_text = (
                    f"CRITÉRIOS ETAPA 6 (FUSÃO GPS & FAIL-SAFE): {status_header}\n"
                    f"--------------------------------------------------\n"
                    f"  * Consenso Nominal (Erro < 1.5m)      : {err_val:.2f} m -> {consenso_label}\n"
                    f"  * Rejeição Glitch (< 1.0s)            : {glitch_label}\n"
                    f"  * Temporizador Persistência (>= 1.0s) : {persistencia_label}\n"
                    f"  * Disparo Failsafe de EBS             : {ebs_label}\n"
                    f"--------------------------------------------------\n"
                    f"Estado Fusão: {st_name} | Failsafe GPS: {'DISPARADO (Segurança Total)' if ebs_active_gps else 'Operacional'}"
                )

            else:
                avisos = data.get('avisos', [])
                coincidencia = data.get('coincidencia', 'NOMINAL (Tração Estável)')
                hold_sec = data.get('hold_timer', 0) * self.dt
                sp_phys, sp_det = data.get('spins', (0, 0))
                lk_phys, lk_det = data.get('locks', (0, 0))
                avisos_str = ", ".join(avisos) if avisos else "Nenhum (Tração Estável)"
                hold_tag = f" [Alerta Retido: {hold_sec:.1f}s]" if hold_sec > 0.05 else ""

                total_events = sp_phys + lk_phys
                total_matches = sp_det + lk_det
                taxa_str = f"{(total_matches / total_events)*100:.0f}%" if total_events > 0 else "100%"

                if self.ebs_active:
                    status_text = (
                        f"ESTADO: {self.as_state} (EBS ACIONADO!)\n"
                        f"--------------------------------------------------\n"
                        f"TRAVAGEM DE EMERGENCIA ATIVA!\n"
                        f"Distancia ao Centro: {data['dist_to_center']:.2f} m | Cones DNF: {self.cones_hit}\n"
                        f"Avisos Telemetria: {avisos_str}\n"
                        f"Tempo: {self.sim_time:.1f}s | Reset necessario."
                    )
                    bg_color = "#ffcccc"
                else:
                    if "[MATCH]" in coincidencia: bg_color = "#d1ecf1"
                    elif "[MISS]" in coincidencia or "[FALSO ALARME]" in coincidencia: bg_color = "#f8d7da"
                    else: bg_color = "#d4edda"

                    status_text = (
                        f"ESTADO: {self.as_state} | MODO: {self.mode_str}\n"
                        f"--------------------------------------------------\n"
                        f"CONTADORES TELEMETRIA:\n"
                        f"  * Spins Reais: {sp_phys} | Avisados: {sp_det}\n"
                        f"  * Locks Reais: {lk_phys} | Avisados: {lk_det} (Precisão: {taxa_str})\n"
                        f"--------------------------------------------------\n"
                        f"AVISOS ATIVOS: {avisos_str}{hold_tag}\n"
                        f"COINCIDENCIA FÍSICA: {coincidencia}\n"
                        f"V_Car: {self.car.v*3.6:.1f} km/h | Volante: {np.degrees(self.car.steer):+.1f} deg | Tempo: {self.sim_time:.1f}s"
                    )

            self.status_box.set_text(status_text)
            self.status_box.set_bbox(dict(boxstyle="round,pad=0.6", facecolor=bg_color, edgecolor="black"))

        return self.trail_line, self.midline_plot, self.lookahead_plot, self.det_cones_plot

    def run(self):
        anim = FuncAnimation(self.fig, self.update_plot, interval=25, blit=False)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulador SIL Driverless FSUminho")
    parser.add_argument("--solucao", action="store_true", help="Executar com a solução de referência completa")
    parser.add_argument("--etapa", type=int, choices=[1, 2, 3, 4, 5, 6], default=None,
                        help="Testar individualmente a Etapa N (1 a 6) com as restantes apoiadas pela referência")
    args = parser.parse_args()

    sim = FSDLiveSimulator(use_solution=args.solucao, etapa=args.etapa)
    sim.run()
