"""
=============================================================================
FSUminho AS / ESW Challenge - Validador de Estrutura e Conformidade de Dados
=============================================================================
Corre uma bateria de testes unitários rápidos nas 5 tarefas para garantir
que as funções dos participantes têm a assinatura, formatos de entrada e
de saída exatamente como o simulador requer antes de executar a corrida!

Uso:
  python verificar_estrutura.py               # Testa desafio_aluno.py
  python verificar_estrutura.py --solucao     # Testa solucao.py
=============================================================================
"""

import sys
import os
import argparse
import numpy as np

# Configuração automática de caminhos para a arquitetura do projeto
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
for _path in [_ROOT_DIR, os.path.join(_ROOT_DIR, "candidatos"), os.path.join(_ROOT_DIR, "core"), os.path.join(_ROOT_DIR, "organizador"), _CURRENT_DIR]:
    if os.path.exists(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from contrato_dados import CarState, ASState
from gerador_pista import TrackGenerator

# Garantir encoding UTF-8 no Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Cores ANSI para o terminal
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def log_pass(msg):
    print(f"  {GREEN}[PASS]{RESET} {msg}")

def log_fail(msg, err=""):
    print(f"  {RED}[FAIL]{RESET} {msg}")
    if err:
        print(f"         {YELLOW}Motivo:{RESET} {err}")


def run_checks(module_name: str):
    print(f"\n{BOLD}{CYAN}================================================================={RESET}")
    print(f"{BOLD}  VERIFICAÇÃO DE ESTRUTURA E CONTRATO DE DADOS - FSUMINHO AS/ESW{RESET}")
    print(f"  Módulo em teste: {BOLD}{module_name}{RESET}")
    print(f"{BOLD}{CYAN}================================================================={RESET}\n")

    try:
        mod = __import__(module_name)
    except Exception as e:
        print(f"{RED}[ERRO CRÍTICO]{RESET} Não foi possível importar '{module_name}': {e}")
        return False

    total_tasks = 5
    passed_tasks = 0

    # Inicializar gerador de pista para testes de cenários reais
    track_gen = TrackGenerator()
    gt = track_gen.get_ground_truth_track()
    gt_mid = np.array(gt['midline'])

    # -------------------------------------------------------------------------
    # TESTE 1: Filtrar e classificar cones (Contrato + 4 Cenários de Pista)
    # -------------------------------------------------------------------------
    print(f"{BOLD}1. [AS - Perceção] filtrar_e_classificar_cones:{RESET}")
    t1_pass = True
    if not hasattr(mod, "filtrar_e_classificar_cones"):
        log_fail("Função 'filtrar_e_classificar_cones' não encontrada.")
        t1_pass = False
    else:
        # 1.1 Contrato básico com ruído e desconhecidos
        sample_input = [
            {'x': 6.0, 'y': 4.0, 'color': 'blue', 'confidence': 0.95},   # Azul distante (dist ~ 7.21)
            {'x': 2.0, 'y': 1.5, 'color': 'blue', 'confidence': 0.90},   # Azul perto (dist ~ 2.50)
            {'x': 8.0, 'y': 2.0, 'color': 'blue', 'confidence': 0.40},   # Descartar (conf < 0.60)
            {'x': 3.0, 'y': 0.0, 'color': 'unknown', 'confidence': 0.90},# Descartar (unknown)
            {'x': 7.0, 'y': -3.0, 'color': 'yellow', 'confidence': 0.75}, # Amarelo distante (dist ~ 7.62)
            {'x': 4.0, 'y': -1.0, 'color': 'yellow', 'confidence': 0.88}  # Amarelo perto (dist ~ 4.12)
        ]
        try:
            res = mod.filtrar_e_classificar_cones(sample_input)
            if not isinstance(res, (tuple, list)) or len(res) != 2:
                log_fail("O retorno deve ser um tuplo (cones_azuis, cones_amarelos).")
                t1_pass = False
            else:
                azuis, amarelos = res
                if not isinstance(azuis, np.ndarray) or not isinstance(amarelos, np.ndarray):
                    log_fail("Tanto cones_azuis como cones_amarelos devem ser numpy.ndarray!")
                    t1_pass = False
                elif azuis.ndim != 2 or azuis.shape[1] != 2 or amarelos.ndim != 2 or amarelos.shape[1] != 2:
                    log_fail(f"Formato inválido. Esperado shape (N, 2), recebido azuis {azuis.shape} e amarelos {amarelos.shape}.")
                    t1_pass = False
                elif len(azuis) != 2 or len(amarelos) != 2:
                    log_fail("Filtragem de cones incorreta. Dica: Confirma os critérios de descarte (confiança mínima e cores aceites no regulamento).")
                    t1_pass = False
                else:
                    dist_azuis = np.hypot(azuis[:, 0], azuis[:, 1])
                    dist_amarelos = np.hypot(amarelos[:, 0], amarelos[:, 1])
                    if np.any(np.diff(dist_azuis) < -1e-5) or np.any(np.diff(dist_amarelos) < -1e-5):
                        log_fail("Ordenação de proximidade incorreta. Dica: Cada lista deve vir ordenada do cone mais próximo para o mais distante do carro.")
                        t1_pass = False
                    else:
                        log_pass("Contrato de tipos, filtro de ruído e ordenação vetorizada validados.")

            # 1.2 Bateria Extensiva nos 4 Cenários de Pista (Reta, Curva, Chicane, Hairpin)
            if t1_pass:
                cenarios_pista_t1 = [
                    {"nome": "Reta c/ Ruído", "x": 0.0, "y": 5.0, "yaw": np.radians(85)},
                    {"nome": "Curva Fechada", "x": 10.0, "y": 30.0, "yaw": np.radians(70)},
                    {"nome": "Chicane Técnica", "x": 42.0, "y": 37.0, "yaw": np.radians(-45)},
                    {"nome": "Hairpin Cego", "x": 50.0, "y": 48.0, "yaw": np.radians(-65)},
                ]
                cenarios_ok = 0
                for sc in cenarios_pista_t1:
                    raw_dets = track_gen.simulate_sensor_detections(
                        sc["x"], sc["y"], sc["yaw"], max_range=16.0, fov_deg=120.0, add_noise=True, extra_noise=True
                    )
                    az_res, am_res = mod.filtrar_e_classificar_cones(raw_dets)
                    
                    # Cones esperados de referência
                    exp_b = set((round(d['x'], 2), round(d['y'], 2)) for d in raw_dets if d.get('color') == 'blue' and d.get('confidence', 0) >= 0.60)
                    exp_y = set((round(d['x'], 2), round(d['y'], 2)) for d in raw_dets if d.get('color') == 'yellow' and d.get('confidence', 0) >= 0.60)
                    
                    act_b = set((round(p[0], 2), round(p[1], 2)) for p in az_res) if len(az_res) > 0 else set()
                    act_y = set((round(p[0], 2), round(p[1], 2)) for p in am_res) if len(am_res) > 0 else set()
                    
                    # Verificar se não aceitou ruído e pegou todos os válidos
                    falsos_positivos = (act_b - exp_b) | (act_y - exp_y)
                    omissos = (exp_b - act_b) | (exp_y - act_y)
                    
                    # Verificar ordenação
                    ordem_ok = True
                    if len(az_res) > 1:
                        db = np.hypot(az_res[:, 0], az_res[:, 1])
                        if np.any(np.diff(db) < -1e-4): ordem_ok = False
                    if len(am_res) > 1:
                        dy = np.hypot(am_res[:, 0], am_res[:, 1])
                        if np.any(np.diff(dy) < -1e-4): ordem_ok = False
                        
                    if len(falsos_positivos) == 0 and len(omissos) == 0 and ordem_ok:
                        cenarios_ok += 1
                        
                if cenarios_ok == len(cenarios_pista_t1):
                    log_pass(f"Bateria Extensiva: 4/4 Cenários de Pista Aprovados (100% cones legítimos, 0 falsos positivos e ordenação OK).")
                else:
                    log_fail("Falha em cenários de pista com ruído de sensor. Dica: Garante que ruídos e reflexões são eliminados e as cores estão corretamente separadas.")
                    t1_pass = False

        except NotImplementedError:
            log_fail("Função ainda não implementada (NotImplementedError).")
            t1_pass = False
        except Exception as e:
            log_fail("Erro durante a execução.", str(e))
            t1_pass = False

    if t1_pass: passed_tasks += 1

    # -------------------------------------------------------------------------
    # TESTE 2: Calcular Midline (Contrato + Edge Cases + 4 Cenários de Pista)
    # -------------------------------------------------------------------------
    print(f"\n{BOLD}2. [AS - Planeamento] calcular_midline_trajetoria:{RESET}")
    t2_pass = True
    if not hasattr(mod, "calcular_midline_trajetoria"):
        log_fail("Função 'calcular_midline_trajetoria' não encontrada.")
        t2_pass = False
    else:
        try:
            # 2.1 Caso limite de segurança: listas vazias
            empty_res = mod.calcular_midline_trajetoria(np.empty((0, 2)), np.empty((0, 2)))
            if not isinstance(empty_res, np.ndarray):
                log_fail("Com entradas vazias, deve retornar um numpy.ndarray!")
                t2_pass = False
            elif len(empty_res) != 0:
                log_fail("Com entradas vazias, deve retornar array vazio np.empty((0, 2))!")
                t2_pass = False
            else:
                log_pass("Proteção Fail-Safe em lista vazia validada (retorna np.empty((0, 2))).")

            # 2.2 Caso Base Nominal
            sample_azuis = np.array([[0.0, 1.6], [5.0, 1.6], [10.0, 1.6]])
            sample_amarelos = np.array([[0.0, -1.6], [5.0, -1.6], [10.0, -1.6]])
            res_nominal = mod.calcular_midline_trajetoria(sample_azuis, sample_amarelos)
            if not isinstance(res_nominal, np.ndarray) or res_nominal.ndim != 2 or res_nominal.shape[1] != 2:
                log_fail("Shape de retorno inválido. Esperado (K, 2).")
                t2_pass = False
            elif len(res_nominal) == 0:
                log_fail("Retornou vazio com cones perfeitamente emparelhados!")
                t2_pass = False
            elif abs(res_nominal[0, 1]) > 0.5:
                log_fail("Cálculo do corredor central impreciso. Dica: Numa pista reta e simétrica, a midline deve passar exatamente a meio dos pares de cones.")
                t2_pass = False
            else:
                log_pass(f"Caso Base Nominal aprovado ({len(res_nominal)} waypoints centralizados).")

            # 2.3 Bateria Extensiva nos 4 Cenários de Planeamento (Nominal, Curva Cega Esq/Dir, Chicane)
            if t2_pass:
                cenarios_t2 = [
                    {"nome": "Cenário 1 (Nominal)", "x": 0.0, "y": 15.0, "yaw": np.radians(65), "drop": None, "min_pts": 3, "max_dev": 1.0},
                    {"nome": "Cenário 2 (Curva Cega Esq - Sem Azuis)", "x": 10.0, "y": 35.0, "yaw": np.radians(70), "drop": 'blue', "min_pts": 3, "max_dev": 1.5},
                    {"nome": "Cenário 3 (Curva Cega Dir - Sem Amarelos)", "x": 50.0, "y": 48.0, "yaw": np.radians(-65), "drop": 'yellow', "min_pts": 3, "max_dev": 1.5},
                    {"nome": "Cenário 4 (Chicane em S)", "x": 42.0, "y": 37.0, "yaw": np.radians(-45), "drop": None, "min_pts": 4, "max_dev": 1.0},
                ]
                c2_ok = 0
                for sc in cenarios_t2:
                    raw = track_gen.simulate_sensor_detections(sc["x"], sc["y"], sc["yaw"], max_range=14.0, fov_deg=110.0, add_noise=False, drop_side=sc["drop"])
                    az_in = np.array([[d['x'], d['y']] for d in raw if d['color'] == 'blue'])
                    am_in = np.array([[d['x'], d['y']] for d in raw if d['color'] == 'yellow'])
                    if len(az_in) == 0: az_in = np.empty((0, 2))
                    if len(am_in) == 0: am_in = np.empty((0, 2))

                    mid_res = mod.calcular_midline_trajetoria(az_in, am_in)
                    if isinstance(mid_res, np.ndarray) and mid_res.ndim == 2 and len(mid_res) >= sc["min_pts"]:
                        desvios = [np.min(np.hypot(gt_mid[:, 0] - pt[0], gt_mid[:, 1] - pt[1])) for pt in mid_res]
                        if max(desvios) <= sc["max_dev"]:
                            c2_ok += 1
                        else:
                            log_fail(f"{sc['nome']}: A trajetória estimada afasta-se do centro da pista. Dica: Valida geometricamente os pares de cones para evitar emparelhar cones de troços opostos.")
                    else:
                        log_fail(f"{sc['nome']}: Trajetória incompleta. Dica: Numa curva cega onde um dos lados não é visível pelo sensor, estima a linha central com base no lado visível e na largura da pista.")

                if c2_ok == len(cenarios_t2):
                    log_pass("Bateria Extensiva: 4/4 Cenários de Planeamento Aprovados (Nominal, Fallback Curvas Cegas e Chicane S).")
                else:
                    t2_pass = False

        except NotImplementedError:
            log_fail("Função ainda não implementada (NotImplementedError).")
            t2_pass = False
        except Exception as e:
            log_fail("Erro durante a execução.", str(e))
            t2_pass = False

    if t2_pass: passed_tasks += 1

    # -------------------------------------------------------------------------
    # TESTE 3: Pure Pursuit & Pedais (5 Casos Geométricos Rigorosos)
    # -------------------------------------------------------------------------
    print(f"\n{BOLD}3. [AS - Controlo] controlador_pure_pursuit:{RESET}")
    t3_pass = True
    if not hasattr(mod, "controlador_pure_pursuit"):
        log_fail("Função 'controlador_pure_pursuit' não encontrada.")
        t3_pass = False
    else:
        try:
            # 3.1 Caso 1: Reta Perfeita (Steer ~ 0.0 e aceleração APPS >= 20%)
            car_reta = CarState(x=0.0, y=0.0, yaw=0.0, v=8.0, steer=0.0)
            wps_reta = np.array([[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0]])
            res1 = mod.controlador_pure_pursuit(car_reta, wps_reta, lookahead_distance=4.0, wheelbase=1.55)
            s1, a1, b1 = float(res1[0]), float(res1[1]), float(res1[2]) if isinstance(res1, (tuple, list)) else (float(res1), 50.0, 0.0)

            if abs(s1) >= 0.05:
                log_fail("Caso 1 (Reta): Volante desalinhado na reta. Dica: Quando a pista segue em frente sem curvatura, o ângulo de direção deve tender para zero.")
                t3_pass = False
            elif a1 < 20.0:
                log_fail("Caso 1 (Reta): Aceleração nula ou insuficiente na reta. Dica: Em linha reta deves acelerar forte para ganhar tempo de volta sem perigo de despiste.")
                t3_pass = False
            else:
                log_pass(f"Caso 1 (Reta): Volante centrado ({np.degrees(s1):.1f}°) e aceleração forte (APPS {a1:.1f}%).")

            # 3.2 Caso 2: Curva à Esquerda (Steer positivo > 0.10 rad)
            car_esq = CarState(x=0.0, y=0.0, yaw=0.0, v=7.0, steer=0.0)
            wps_esq = np.array([[2.0, 0.5], [4.0, 1.8], [6.0, 3.5], [8.0, 5.5]])
            res2 = mod.controlador_pure_pursuit(car_esq, wps_esq, lookahead_distance=4.0, wheelbase=1.55)
            s2, a2, b2 = float(res2[0]), float(res2[1]), float(res2[2]) if isinstance(res2, (tuple, list)) else (float(res2), 50.0, 0.0)
            if s2 <= 0.10:
                log_fail("Caso 2 (Curva Esq): Sentido de direção incorreto. Dica: Numa curva para a esquerda, o volante deve virar para a esquerda (ângulo positivo no referencial do carro).")
                t3_pass = False
            else:
                log_pass(f"Caso 2 (Curva Esq): Viragem positiva validada (Steer={s2:.3f} rad / +{np.degrees(s2):.1f}°).")

            # 3.3 Caso 3: Curva à Direita (Steer negativo < -0.10 rad)
            car_dir = CarState(x=0.0, y=0.0, yaw=0.0, v=7.0, steer=0.0)
            wps_dir = np.array([[2.0, -0.5], [4.0, -1.8], [6.0, -3.5], [8.0, -5.5]])
            res3 = mod.controlador_pure_pursuit(car_dir, wps_dir, lookahead_distance=4.0, wheelbase=1.55)
            s3, a3, b3 = float(res3[0]), float(res3[1]), float(res3[2]) if isinstance(res3, (tuple, list)) else (float(res3), 50.0, 0.0)
            if s3 >= -0.10:
                log_fail("Caso 3 (Curva Dir): Sentido de direção incorreto. Dica: Numa curva para a direita, o volante deve virar para a direita (ângulo negativo no referencial do carro).")
                t3_pass = False
            else:
                log_pass(f"Caso 3 (Curva Dir): Viragem negativa validada (Steer={s3:.3f} rad / {np.degrees(s3):.1f}°).")

            # 3.4 Caso 4: Saturação Mecânica / Clamp (Alvo Perpendicular a +-0.50 rad)
            car_sat = CarState(x=0.0, y=0.0, yaw=0.0, v=5.0, steer=0.0)
            wps_sat = np.array([[1.0, 4.0], [2.0, 8.0], [3.0, 12.0]])
            res4 = mod.controlador_pure_pursuit(car_sat, wps_sat, lookahead_distance=4.0, wheelbase=1.55)
            s4, a4, b4 = float(res4[0]), float(res4[1]), float(res4[2]) if isinstance(res4, (tuple, list)) else (float(res4), 50.0, 0.0)
            if abs(s4) > 0.505:
                log_fail("Caso 4 (Saturação): O atuador ultrapassou o curso mecânico máximo. Dica: Deves limitar (clamp) o ângulo para respeitar os batentes físicos da cremalheira.")
                t3_pass = False
            elif abs(abs(s4) - 0.50) > 0.02:
                log_fail("Caso 4 (Saturação): Em curva extremamente fechada a direção não atingiu o curso total. Dica: Revê o cálculo de Pure Pursuit para ângulos elevados.")
                t3_pass = False
            else:
                log_pass(f"Caso 4 (Saturação): Saturação física mecânica validada a +-0.50 rad ({np.degrees(s4):.1f}°).")

            # 3.5 Caso 5: Fail-Safe em Trajetória Vazia
            res5 = mod.controlador_pure_pursuit(car_reta, np.empty((0, 2)), lookahead_distance=4.0, wheelbase=1.55)
            s5, a5, b5 = float(res5[0]), float(res5[1]), float(res5[2]) if isinstance(res5, (tuple, list)) else (float(res5), 0.0, 40.0)
            if abs(s5) > 0.10:
                log_fail("Caso 5 (Fail-Safe Vazio): Comportamento de volante inseguro sem waypoints. Dica: Se não existirem waypoints à frente, o volante deve permanecer centrado.")
                t3_pass = False
            elif b5 < 10.0:
                log_fail("Caso 5 (Fail-Safe Vazio): Travagem ausente em perda de waypoints. Dica: Em perda de visão da pista, o veículo deve aplicar travagem de segurança para se imobilizar.")
                t3_pass = False
            else:
                log_pass(f"Caso 5 (Fail-Safe Vazio): Proteção de travagem de emergência confirmada (Brake={b5:.1f} bar).")

            if t3_pass:
                log_pass("Bateria Extensiva: 5/5 Casos de Controlo e Pedais 100% Aprovados!")

        except NotImplementedError:
            log_fail("Função ainda não implementada (NotImplementedError).")
            t3_pass = False
        except Exception as e:
            log_fail("Erro durante a execução.", str(e))
            t3_pass = False

    if t3_pass: passed_tasks += 1

    # -------------------------------------------------------------------------
    # TESTE 4: Dinâmica e Escorregamento
    # -------------------------------------------------------------------------
    print(f"\n{BOLD}4. [ESW - Telemetria] analisar_dinamica_veiculo:{RESET}")
    t4_pass = True
    if not hasattr(mod, "analisar_dinamica_veiculo"):
        log_fail("Função 'analisar_dinamica_veiculo' não encontrada.")
        t4_pass = False
    else:
        telemetry_test = {
            'est_car_velocity': 10.0,
            'fl_wheel_speed': 10.0,
            'fr_wheel_speed': 10.0,
            'rl_wheel_speed': 12.5,  # Slip traseiro = (12.5-10)/10 = 0.25 (> 0.15)
            'rr_wheel_speed': 12.0,
            'accel_x': 2.5,
            'accel_y': 14.7,         # 14.7 / 9.81 ~ 1.5 G
            'apps': 60.0,            # Acelerador ativo (> 10%) -> Deve detetar Wheel Spin!
            'brake_pressure': 0.0
        }
        try:
            res = mod.analisar_dinamica_veiculo(telemetry_test)
            expected_keys = {'rl_slip', 'rr_slip', 'wheel_spin_detected', 'wheel_lock_detected', 'lateral_g', 'avisos'}
            if not isinstance(res, dict):
                log_fail(f"O retorno deve ser um dict, recebido {type(res)}.")
                t4_pass = False
            elif not expected_keys.issubset(res.keys()):
                log_fail("Dicionário de telemetria incompleto. Dica: Consulta o enunciado para verificar todas as chaves obrigatórias que devem ser retornadas.")
                t4_pass = False
            elif not isinstance(res.get('avisos'), (list, tuple)):
                log_fail(f"A chave 'avisos' deve ser uma lista de strings, recebido {type(res.get('avisos'))}.")
                t4_pass = False
            elif "AVISO_WHEEL_SPIN" not in res.get('avisos', []):
                log_fail("Alarme em falta na lista de avisos. Dica: Quando é detetado Wheel Spin sob aceleração, deves adicionar 'AVISO_WHEEL_SPIN' à lista de avisos.")
                t4_pass = False
            elif not res.get('wheel_spin_detected', False):
                log_fail("Deteção de Wheel Spin incorreta. Dica: Verifica a condição de aceleração e o limiar de patinagem das rodas traseiras motrizes.")
                t4_pass = False
            elif abs(res.get('lateral_g', 0.0) - 1.5) > 0.1:
                log_fail("Cálculo de aceleração lateral em G impreciso. Dica: Converte a aceleração lateral da IMU em m/s² para múltiplos da aceleração da gravidade terrestre.")
                t4_pass = False
            else:
                log_pass("Contrato base de telemetria, Slip Ratio e deteção de Wheel Spin validados.")

            # 4.2 Teste do Filtro de Debounce Temporal de Wheel Lock (mínimo 3 amostras)
            if t4_pass:
                lock_packet = {
                    'est_car_velocity': 14.0,
                    'fl_wheel_speed': 3.0,   # Slip FL = (3 - 14)/14 = -0.78 (< -0.20)
                    'fr_wheel_speed': 14.0,
                    'rl_wheel_speed': 14.0,
                    'rr_wheel_speed': 14.0,
                    'accel_x': -6.0,
                    'accel_y': 0.0,
                    'apps': 0.0,
                    'brake_pressure': 40.0
                }
                normal_packet = {
                    'est_car_velocity': 14.0,
                    'fl_wheel_speed': 14.0,
                    'fr_wheel_speed': 14.0,
                    'rl_wheel_speed': 14.0,
                    'rr_wheel_speed': 14.0,
                    'accel_x': 0.0,
                    'accel_y': 0.0,
                    'apps': 0.0,
                    'brake_pressure': 0.0
                }
                # Reset
                mod.analisar_dinamica_veiculo(normal_packet)

                # Amostra 1 (ruído isolado): não deve disparar alarme
                res_l1 = mod.analisar_dinamica_veiculo(lock_packet)
                # Amostra 2
                res_l2 = mod.analisar_dinamica_veiculo(lock_packet)
                # Amostra 3 (confirmação persistente): deve disparar
                res_l3 = mod.analisar_dinamica_veiculo(lock_packet)

                if res_l1.get('wheel_lock_detected') or "AVISO_WHEEL_LOCK" in res_l1.get('avisos', []):
                    log_fail("Filtro de Debounce falhou: um ruído pontual de 1 amostra ativou o alarme. Dica: O alarme só deve disparar se o bloqueio persistir durante pelo menos 3 amostras consecutivas.")
                    t4_pass = False
                elif not res_l3.get('wheel_lock_detected') or "AVISO_WHEEL_LOCK" not in res_l3.get('avisos', []):
                    log_fail("Deteção de Wheel Lock falhou após persistência temporal. Dica: Após 3 amostras consecutivas com escorregamento e travão ativos, 'wheel_lock_detected' deve ser True com 'AVISO_WHEEL_LOCK'.")
                    t4_pass = False
                else:
                    log_pass("Filtro de Debounce temporal validado (rejeita ruído isolado e confirma bloqueio à 3ª amostra).")

        except NotImplementedError:
            log_fail("Função ainda não implementada (NotImplementedError).")
            t4_pass = False
        except Exception as e:
            log_fail("Erro durante a execução.", str(e))
            t4_pass = False

    if t4_pass: passed_tasks += 1

    # -------------------------------------------------------------------------
    # TESTE 5: Safety State Machine e EBS
    # -------------------------------------------------------------------------
    print(f"\n{BOLD}5. [ESW - Safety & EBS] safety_state_machine_ebs:{RESET}")
    t5_pass = True
    if not hasattr(mod, "safety_state_machine_ebs"):
        log_fail("Função 'safety_state_machine_ebs' não encontrada.")
        t5_pass = False
    else:
        try:
            # Caso A: Partida normal
            st1, ebs1 = mod.safety_state_machine_ebs("AS_READY", distance_to_midline=0.2, cones_seen=10, mission_start_cmd=True)
            # Caso B: Em pista normal
            st2, ebs2 = mod.safety_state_machine_ebs("AS_DRIVING", distance_to_midline=0.4, cones_seen=8, mission_start_cmd=False)
            # Caso C: Saída de pista crítica (Off-track > 2.0m)
            st3, ebs3 = mod.safety_state_machine_ebs("AS_DRIVING", distance_to_midline=2.4, cones_seen=6, mission_start_cmd=False)
            # Caso D: Perda total de cones (0 cones visíveis)
            st4, ebs4 = mod.safety_state_machine_ebs("AS_DRIVING", distance_to_midline=0.3, cones_seen=0, mission_start_cmd=False)
            
            if st1 != "AS_DRIVING" or ebs1 != False:
                log_fail("Transição de largada incorreta. Dica: Quando o carro está em AS_READY e recebe o comando de início de missão, deve transitar para AS_DRIVING.")
                t5_pass = False
            elif st2 != "AS_DRIVING" or ebs2 != False:
                log_fail("Condução nominal em pista incorreta. Dica: Dentro dos limites da pista e com cones visíveis, o monolugar deve manter-se em AS_DRIVING sem acionar EBS.")
                t5_pass = False
            elif st3 != "AS_EMERGENCY" or ebs3 != True:
                log_fail("Gatilho de segurança Off-Track falhou. Dica: Se a distância ao corredor central exceder o limite de segurança de pista, deve transitar de imediato para AS_EMERGENCY com EBS ativo.")
                t5_pass = False
            elif st4 != "AS_EMERGENCY" or ebs4 != True:
                log_fail("Gatilho de perda de visão falhou. Dica: Se o monolugar deixar de ver cones durante a condução, a máquina de estados deve cortar imediatamente para AS_EMERGENCY com EBS acionado.")
                t5_pass = False
            else:
                log_pass("Transições de estado e gatilhos de EBS (Off-track e perda de cones) 100% verificados.")
        except NotImplementedError:
            log_fail("Função ainda não implementada (NotImplementedError).")
            t5_pass = False
        except Exception as e:
            log_fail("Erro durante a execução.", str(e))
            t5_pass = False

    if t5_pass: passed_tasks += 1

    # -------------------------------------------------------------------------
    # TESTE 6: [BÓNUS OPCIONAL] Supervisão de Fusão Sensorial & Fail-Safe GPS
    # -------------------------------------------------------------------------
    bonus_pass = False
    print(f"\n{BOLD}6. [ESW - BÓNUS OPCIONAL] supervisionar_fusao_gps:{RESET}")
    if hasattr(mod, "supervisionar_fusao_gps"):
        try:
            if hasattr(mod, "_tempo_divergencia_gps"):
                mod._tempo_divergencia_gps = 0.0
            # Teste 1: Consenso nominal (distância <= 1.5m)
            r1 = mod.supervisionar_fusao_gps((10.0, 5.0), (10.1, 5.1), dt=0.05)
            # Teste 2: Glitch transitório (distância = 2.5m > 1.5m, 1 amostra dt=0.05s < 1.0s)
            r2 = mod.supervisionar_fusao_gps((10.0, 5.0), (12.5, 5.0), dt=0.05)
            # Teste 3: Divergência sustentada por mais de 1.0s (20 amostras x 0.05s = 1.05s)
            for _ in range(20):
                r3 = mod.supervisionar_fusao_gps((10.0, 5.0), (12.5, 5.0), dt=0.05)
            # Teste 4: Recuperação de consenso
            r4 = mod.supervisionar_fusao_gps((10.0, 5.0), (10.1, 5.0), dt=0.05)

            expected_keys = {'erro_distancia', 'tempo_divergencia', 'estado_fusao', 'ebs_trigger', 'avisos'}
            if not isinstance(r1, dict) or not expected_keys.issubset(r1.keys()):
                log_fail("Retorno da tarefa bónus inválido ou chaves em falta.")
            elif r1.get('estado_fusao') != "FUSION_OK" or r1.get('ebs_trigger') != False:
                log_fail("Consenso nominal falhou. Dica: Para distâncias <= 1.5m, o estado deve ser 'FUSION_OK' com ebs_trigger=False.")
            elif r2.get('estado_fusao') != "RETRY_RESET" or r2.get('ebs_trigger') != False or "AVISO_GPS_DRIFT" not in r2.get('avisos', []):
                log_fail("Filtragem de glitch transitório falhou. Dica: Para divergência < 1.0s, o estado deve ser 'RETRY_RESET' com 'AVISO_GPS_DRIFT' sem disparar EBS.")
            elif r3.get('estado_fusao') != "SENSOR_FUSION_LOST" or r3.get('ebs_trigger') != True or "AVISO_FUSION_FAULT" not in r3.get('avisos', []):
                log_fail("Gatilho de persistência temporal falhou. Dica: Após 1.0s sustentado de erro > 1.5m, deve transitar para 'SENSOR_FUSION_LOST' com ebs_trigger=True.")
            elif r4.get('estado_fusao') != "FUSION_OK" or r4.get('tempo_divergencia') != 0.0:
                log_fail("Recuperação de consenso falhou. Dica: Quando os sensores voltam a concordar, o temporizador de divergência deve reiniciar a 0.0.")
            else:
                log_pass("Fusão Sensorial GPS/Odometria, Limiar de Persistência (1.0s) e Failsafe EBS 100% validados!")
                bonus_pass = True
        except NotImplementedError:
            print(f"  {CYAN}[INFO] Tarefa Bónus não implementada (opcional, não afeta a aprovação obrigatória).{RESET}")
        except Exception as e:
            log_fail("Erro durante a execução da tarefa bónus.", str(e))
    else:
        print(f"  {CYAN}[INFO] Tarefa Bónus não encontrada (opcional).{RESET}")

    # -------------------------------------------------------------------------
    # RESUMO FINAL
    # -------------------------------------------------------------------------
    print(f"\n{BOLD}{CYAN}-----------------------------------------------------------------{RESET}")
    print(f"{BOLD}RESULTADO FINAL:{RESET} {passed_tasks}/{total_tasks} Tarefas Obrigatórias Aprovadas.")
    if bonus_pass:
        print(f"{GREEN}{BOLD}>>> PARABENS DESTAQUE! 5/5 Tarefas Obrigatórias + BÓNUS DE FUSÃO SENSORIAL APROVADO! (110%){RESET}")
        print(f"Podes agora validar os relatórios gráficos estáticos por etapa:")
        print(f"  {CYAN}python tests/avaliar_estatico.py --etapa 1{RESET}  (ou --etapa 2 a 6)")
    elif passed_tasks == total_tasks:
        print(f"{GREEN}{BOLD}>>> PARABENS! A estrutura cumpre 100% dos requisitos de dados obrigatórios.{RESET}")
        print(f"Podes agora validar os relatórios gráficos estáticos por etapa:")
        print(f"  {CYAN}python tests/avaliar_estatico.py --etapa 1{RESET}  (ou --etapa 2 a 6)")
    else:
        print(f"{YELLOW}Ainda existem {total_tasks - passed_tasks} tarefas obrigatórias por concluir ou corrigir.{RESET}")
    print(f"{BOLD}{CYAN}-----------------------------------------------------------------{RESET}\n")
    return passed_tasks == total_tasks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verificador de Estrutura do Desafio FSUminho")
    parser.add_argument("--solucao", action="store_true", help="Testar o ficheiro solucao.py em vez de desafio_aluno.py")
    args = parser.parse_args()

    target = "solucao" if args.solucao else "desafio_aluno"
    success = run_checks(target)
    sys.exit(0 if success else 1)
