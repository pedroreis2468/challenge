"""
=============================================================================
AVALIAÇÃO ESTÁTICA DEDICADA POR ETAPA: FSUMINHO AUTONOMOUS SYSTEMS & ESW
Tempo Limite: 1h30 | 5 Membros de Equipa
=============================================================================
Uso Dedicado por Etapa:
  python avaliar_estatico.py --etapa 1             # Avalia APENAS Tarefa 1 (Perceção pura em 4 cenários)
  python avaliar_estatico.py --etapa 2             # Avalia APENAS Tarefa 2 (Planeamento / Midline e Edge Cases)
  python avaliar_estatico.py --etapa 3             # Avalia APENAS Tarefa 3 (Controlo Pure Pursuit & Ângulos)
  python avaliar_estatico.py --etapa 4             # Avalia APENAS Tarefa 4 (Telemetria 4 Rodas, Slips & Alarmes)
  python avaliar_estatico.py --etapa 5             # Avalia APENAS Tarefa 5 (Safety State Machine & EBS)

Opções Globais:
  --solucao   # Avalia usando a solução gabarito de referência
  --salvar    # Guarda o gráfico em ficheiro PNG em vez de abrir janela
=============================================================================
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from typing import Dict, List, Any

# Configuração automática de caminhos para a arquitetura do projeto
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
for _path in [_ROOT_DIR, os.path.join(_ROOT_DIR, "candidatos"), os.path.join(_ROOT_DIR, "core"), os.path.join(_ROOT_DIR, "organizador"), _CURRENT_DIR]:
    if os.path.exists(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

_BASE_DIR = _ROOT_DIR
from contrato_dados import CarState, ASState
from gerador_pista import TrackGenerator

def _salvar_ou_mostrar(nome_img: str, salvar: bool):
    if salvar:
        out_dir = os.path.join(_BASE_DIR, "assets") if os.path.isdir(os.path.join(_BASE_DIR, "assets")) else _BASE_DIR
        path_salvar = os.path.join(out_dir, nome_img)
        plt.savefig(path_salvar, dpi=160)
        print(f"[IMAGEM GERADA]: '{path_salvar}' guardada com sucesso!")
    else:
        fig = plt.gcf()
        if fig and hasattr(fig, 'canvas') and fig.canvas.manager:
            tb = getattr(fig.canvas.manager, 'toolbar', None)
            if tb and hasattr(tb, 'pan'):
                try:
                    tb.pan()
                except Exception:
                    pass
        plt.show()

_ativar_interatividade = lambda fig, axes: _ativar_zoom_roda(fig, axes)

def _ativar_zoom_roda(fig, axes):
    orig_limits = {ax: (ax.get_xlim(), ax.get_ylim()) for ax in axes}

    def on_scroll(event):
        if event.inaxes is None or event.inaxes not in axes:
            return
        ax = event.inaxes
        scale_factor = 0.80 if event.button == 'up' else 1.25
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return
        new_w = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_h = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        rel_x = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        rel_y = (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])
        ax.set_xlim([xdata - new_w * rel_x, xdata + new_w * (1 - rel_x)])
        ax.set_ylim([ydata - new_h * rel_y, ydata + new_h * (1 - rel_y)])
        fig.canvas.draw_idle()

    def on_dblclick(event):
        if event.inaxes in axes and event.dblclick:
            ax = event.inaxes
            x0, y0 = orig_limits[ax]
            ax.set_xlim(x0)
            ax.set_ylim(y0)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_press_event', on_dblclick)


# =============================================================================
# 1. AVALIAÇÃO ESTÁTICA DEDICADA: ETAPA 1 (PERCEÇÃO PURA)
# =============================================================================
def avaliar_etapa1(modulo, track_gen: TrackGenerator, salvar=False):
    """
    Avaliação Estática Pura da Tarefa 1 (Perceção):
    - Base no Sensor: Cones Azuis, Cones Amarelos e tudo o resto (ruído/artefactos) a Laranja.
    - Decisão do Aluno: VERDE sobre os objetos certos aceites, VERMELHO sobre os objetos errados aceites.
    """
    print("\n" + "="*70)
    print(" AVALIAÇÃO ESTÁTICA DEDICADA - ETAPA 1: AS - PERCEÇÃO DE CONES")
    print("="*70)

    cenarios = [
        {"nome": "Cenário 1: Reta com Ruído e Artefactos", "car_x": 0.0, "car_y": 15.0, "yaw": np.radians(65)},
        {"nome": "Cenário 2: Entrada em Curva Fechada", "car_x": 10.0, "car_y": 35.0, "yaw": np.radians(70)},
        {"nome": "Cenário 3: Chicane Técnica com Reflexões", "car_x": 42.0, "car_y": 37.0, "yaw": np.radians(-45)},
        {"nome": "Cenário 4: Hairpin com Artefactos Físicos", "car_x": 132.0, "car_y": 68.0, "yaw": np.radians(-80)},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("ETAPA 1: AS - PERCEÇÃO E FILTRAGEM DE CONES", 
                 fontsize=14, fontweight='bold', y=0.965)
    axes = axes.flatten()

    total_aprovados = 0

    for i, sc in enumerate(cenarios):
        ax = axes[i]
        car_x, car_y, yaw = sc["car_x"], sc["car_y"], sc["yaw"]
        
        raw_dets = track_gen.simulate_sensor_detections(
            car_x, car_y, yaw, max_range=16.0, fov_deg=120.0, add_noise=True, extra_noise=True
        )

        # Classificação real de referência dos dados no sensor:
        # Cones legítimos da pista: cor blue/yellow e confiança >= 0.60
        # Tudo o resto (ruído de baixa confiança, cor unknown, artefactos): LARANJA
        dets_blue_valid = []
        dets_yellow_valid = []
        dets_laranja = []

        for d in raw_dets:
            conf = d.get('confidence', 0.0)
            cor = d.get('color', '').lower()
            if cor == 'blue' and conf >= 0.60:
                dets_blue_valid.append(d)
            elif cor == 'yellow' and conf >= 0.60:
                dets_yellow_valid.append(d)
            else:
                # Tudo além disso: ruído, baixa confiança, unknown, postes fora de pista
                dets_laranja.append(d)

        total_validos = len(dets_blue_valid) + len(dets_yellow_valid)

        # Executar a função do aluno
        try:
            azuis_aluno, amarelos_aluno = modulo.filtrar_e_classificar_cones(raw_dets)
            if not isinstance(azuis_aluno, np.ndarray):
                azuis_aluno = np.array(azuis_aluno, dtype=float).reshape(-1, 2) if azuis_aluno else np.empty((0, 2))
            if not isinstance(amarelos_aluno, np.ndarray):
                amarelos_aluno = np.array(amarelos_aluno, dtype=float).reshape(-1, 2) if amarelos_aluno else np.empty((0, 2))
        except Exception as e:
            print(f"[ERRO] {sc['nome']}: Função do aluno falhou: {e}")
            azuis_aluno, amarelos_aluno = np.empty((0, 2)), np.empty((0, 2))

        # Coordenadas que o aluno aceitou
        coords_aluno_azuis = set((round(p[0], 2), round(p[1], 2)) for p in azuis_aluno)
        coords_aluno_amarelos = set((round(p[0], 2), round(p[1], 2)) for p in amarelos_aluno)
        todas_coords_aluno = coords_aluno_azuis | coords_aluno_amarelos

        # Avaliar acertos (verde) e erros (vermelho)
        certos_azuis = []
        certos_amarelos = []
        errados_aceites = [] # Deteções laranja ou trocadas que o aluno aceitou por engano

        # Verificar azuis legítimos
        for d in dets_blue_valid:
            c = (round(d['x'], 2), round(d['y'], 2))
            if c in coords_aluno_azuis:
                certos_azuis.append([d['x'], d['y']])
            elif c in coords_aluno_amarelos:
                # Classificou azul como amarelo (trocou cor!)
                errados_aceites.append([d['x'], d['y']])

        # Verificar amarelos legítimos
        for d in dets_yellow_valid:
            c = (round(d['x'], 2), round(d['y'], 2))
            if c in coords_aluno_amarelos:
                certos_amarelos.append([d['x'], d['y']])
            elif c in coords_aluno_azuis:
                # Classificou amarelo como azul (trocou cor!)
                errados_aceites.append([d['x'], d['y']])

        # Verificar objetos laranja (ruído / desconhecido) aceites por erro
        for d in dets_laranja:
            c = (round(d['x'], 2), round(d['y'], 2))
            if c in todas_coords_aluno:
                errados_aceites.append([d['x'], d['y']])

        num_certos = len(certos_azuis) + len(certos_amarelos)
        num_errados = len(errados_aceites)
        total_aluno = len(azuis_aluno) + len(amarelos_aluno)

        # Validar ordenação por distância euclidiana crescente (vetorização)
        ordem_ok = True
        if len(azuis_aluno) > 1:
            d_b = np.hypot(azuis_aluno[:, 0], azuis_aluno[:, 1])
            if np.any(np.diff(d_b) < -1e-4):
                ordem_ok = False
        if len(amarelos_aluno) > 1:
            d_y = np.hypot(amarelos_aluno[:, 0], amarelos_aluno[:, 1])
            if np.any(np.diff(d_y) < -1e-4):
                ordem_ok = False

        if total_aluno == 0:
            status_tag = "[FALHOU: 0 Cones Filtrados]"
            color_title = "#b30000"
            sub_title = f"{status_tag} • Sensor: {total_validos} Cones Válidos ({len(dets_blue_valid)}E/{len(dets_yellow_valid)}D) | {len(dets_laranja)} Ruídos/Artefactos"
        elif num_errados == 0 and num_certos == total_validos and ordem_ok:
            status_tag = "[APROVADO 100%]"
            color_title = "#006622"
            total_aprovados += 1
            sub_title = f"{status_tag} • Cones a Verde ({num_certos}/{total_validos}) | Vetorização & Ordem OK"
        elif not ordem_ok and num_errados == 0 and num_certos == total_validos:
            status_tag = "[FALHOU: ORDENAÇÃO]"
            color_title = "#b30000"
            sub_title = f"{status_tag} • Cones corretos mas NÃO ordenados por proximidade crescente!"
        else:
            status_tag = "[FALHOU]"
            color_title = "#b30000"
            sub_title = f"{status_tag} • Certos: {num_certos}/{total_validos} (Verde) | Erros: {num_errados} (Vermelho)"

        # ---------------- PLOTAGEM LEVE E LIMPA ----------------
        gt = track_gen.get_ground_truth_track()
        
        # Pista apenas local para velocidade máxima
        mid_pts = np.array(gt['midline'])
        mask_mid = (np.abs(mid_pts[:, 0] - car_x) < 30) & (np.abs(mid_pts[:, 1] - car_y) < 30)
        if np.any(mask_mid):
            ax.plot(mid_pts[mask_mid, 0], mid_pts[mask_mid, 1], ':', color='gray', alpha=0.35, label='Midline Pista')

        # FOV do Sensor
        angles = np.linspace(yaw - np.radians(60), yaw + np.radians(60), 20)
        arc_x = car_x + 16.0 * np.cos(angles)
        arc_y = car_y + 16.0 * np.sin(angles)
        wedge_x = [car_x] + list(arc_x) + [car_x]
        wedge_y = [car_y] + list(arc_y) + [car_y]
        ax.fill(wedge_x, wedge_y, color='gold', alpha=0.10, label='FOV Sensor (120°, 16m)')

        # 1. OBJETOS BASE NO SENSOR
        # Cones Azuis no Sensor
        if len(dets_blue_valid) > 0:
            arr = np.array([[d['x'], d['y']] for d in dets_blue_valid])
            ax.plot(arr[:, 0], arr[:, 1], 'o', color='royalblue', markersize=7.5, markeredgecolor='navy', label=f'Cones Azuis ({len(dets_blue_valid)})')
            
        # Cones Amarelos no Sensor
        if len(dets_yellow_valid) > 0:
            arr = np.array([[d['x'], d['y']] for d in dets_yellow_valid])
            ax.plot(arr[:, 0], arr[:, 1], 'o', color='gold', markersize=7.5, markeredgecolor='darkgoldenrod', label=f'Cones Amarelos ({len(dets_yellow_valid)})')

        # Tudo além disso: LARANJA (ruído, artefactos físicos fora de pista, unknown)
        if len(dets_laranja) > 0:
            arr = np.array([[d['x'], d['y']] for d in dets_laranja])
            ax.plot(arr[:, 0], arr[:, 1], 's', color='#ff7f0e', markersize=7.5, markeredgecolor='#8c3b00', label=f'Ruído / Artefactos ({len(dets_laranja)})')

        # 2. RESULTADO DO FILTRO DO ALUNO (POR CIMA DOS OBJETOS)
        # Se estiver CERTO: anel/halo VERDE por cima dos objetos certos
        if len(certos_azuis) > 0:
            arr = np.array(certos_azuis)
            ax.plot(arr[:, 0], arr[:, 1], 'o', markerfacecolor='none', markeredgecolor='lime', 
                    markersize=13, markeredgewidth=2.6, label='Filtro Aluno: CERTO (Verde)')
        if len(certos_amarelos) > 0:
            arr = np.array(certos_amarelos)
            ax.plot(arr[:, 0], arr[:, 1], 'o', markerfacecolor='none', markeredgecolor='lime', 
                    markersize=13, markeredgewidth=2.6)

        # Se estiver ERRADO (aceitou objeto laranja ou trocou cor): marcação a VERMELHO por cima!
        if len(errados_aceites) > 0:
            arr = np.array(errados_aceites)
            ax.plot(arr[:, 0], arr[:, 1], 'x', color='red', markersize=12, markeredgewidth=3.0, 
                    label='Filtro Aluno: ERRADO (Vermelho)')

        # Posição do sensor / carro
        ax.plot(car_x, car_y, '^', color='black', markersize=10, markeredgecolor='white', label='Sensor / Carro')
        ax.arrow(car_x, car_y, 2.5 * np.cos(yaw), 2.5 * np.sin(yaw), head_width=0.8, color='black', alpha=0.8)

        ax.set_xlim(car_x - 14, car_x + 18)
        ax.set_ylim(car_y - 14, car_y + 18)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, linestyle=':', alpha=0.45)

        ax.set_title(f"{sc['nome']}\n{sub_title}", fontsize=9.5, fontweight='bold', color=color_title, pad=8)

        print(f"-> {sc['nome']}:")
        print(f"   Sensor Detetou          : {len(raw_dets)} (Válidos: {total_validos} [Azul: {len(dets_blue_valid)}, Amarelo: {len(dets_yellow_valid)}] | Laranja: {len(dets_laranja)})")
        print(f"   Cones Filtrados Aluno   : {len(azuis_aluno)} Azuis, {len(amarelos_aluno)} Amarelos")
        print(f"   Acertos (Verde)         : {num_certos}/{total_validos}")
        print(f"   Erros Aceites (Vermelho): {num_errados}")
        print(f"   Resultado               : {status_tag}\n")

    # Legenda global no fundo
    leg_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', markeredgecolor='navy', markersize=8, label='Cones Azuis'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', markeredgecolor='darkgoldenrod', markersize=8, label='Cones Amarelos'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#ff7f0e', markeredgecolor='#8c3b00', markersize=8, label='Ruído / Outro Objeto (Laranja)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='lime', markersize=10, markeredgewidth=2.4, label='Filtro Certo (Verde)'),
        plt.Line2D([0], [0], marker='x', color='red', linestyle='None', markersize=9, markeredgewidth=2.4, label='Filtro Errado (Vermelho)'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markeredgecolor='black', markersize=8, label='Carro / Sensor')
    ]
    fig.legend(handles=leg_handles, loc='lower center', ncol=6, fontsize=8.5, frameon=True, 
               framealpha=0.92, edgecolor='#cccccc', bbox_to_anchor=(0.5, 0.02))

    # Margens generosas: ZERO colisão entre títulos e gráficos
    plt.subplots_adjust(top=0.86, bottom=0.08, left=0.06, right=0.95, hspace=0.35, wspace=0.18)

    _ativar_zoom_roda(fig, axes)
    _salvar_ou_mostrar("avaliacao_estatica_etapa1.png", salvar)

    return total_aprovados == len(cenarios)

# =============================================================================
# 2. AVALIAÇÃO ESTÁTICA DEDICADA: ETAPA 2 (PLANEAMENTO / MIDLINE & EDGE CASES)
# =============================================================================
def avaliar_etapa2(modulo, track_gen: TrackGenerator, salvar=False):
    """
    Avaliação Estática Pura da Tarefa 2 (Planeamento):
    Testa o cálculo da linha central (midline) em 4 cenários geométricos desafiantes:
    1. Traçado Nominal (Ambos os lados perfeitamente visíveis)
    2. Curva Cega à Esquerda (Perda total de cones azuis - teste de fallback)
    3. Curva Cega à Direita (Perda total de cones amarelos - teste de fallback)
    4. Chicane S-Curve (Emparelhamento com curvatura invertida)
    """
    print("\n" + "="*70)
    print(" AVALIAÇÃO ESTÁTICA DEDICADA - ETAPA 2: AS - PLANEAMENTO & MIDLINE")
    print("="*70)

    cenarios = [
        {"nome": "Cenário 1: Traçado Nominal (Dois Lados Visíveis)", "car_x": 0.0, "car_y": 15.0, "yaw": np.radians(65), "drop": None},
        {"nome": "Cenário 2: Curva Cega Esq. (Sem Cones Azuis)", "car_x": 10.0, "car_y": 35.0, "yaw": np.radians(70), "drop": 'blue'},
        {"nome": "Cenário 3: Curva Cega Dir. (Sem Cones Amarelos)", "car_x": 50.0, "car_y": 48.0, "yaw": np.radians(-65), "drop": 'yellow'},
        {"nome": "Cenário 4: Chicane em S com Pares Próximos", "car_x": 42.0, "car_y": 37.0, "yaw": np.radians(-45), "drop": None},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("ETAPA 2: AS - PLANEAMENTO DA LINHA CENTRAL (MIDLINE)", 
                 fontsize=14, fontweight='bold', y=0.965)
    axes = axes.flatten()

    total_aprovados = 0

    for i, sc in enumerate(cenarios):
        ax = axes[i]
        car_x, car_y, yaw = sc["car_x"], sc["car_y"], sc["yaw"]
        
        # Gerar deteções ground-truth filtradas
        raw_dets = track_gen.simulate_sensor_detections(
            car_x, car_y, yaw, max_range=14.0, fov_deg=110.0, add_noise=False, drop_side=sc["drop"]
        )
        
        # Cones que entram para a Tarefa 2
        azuis_in = np.array([[d['x'], d['y']] for d in raw_dets if d['color'] == 'blue'])
        amarelos_in = np.array([[d['x'], d['y']] for d in raw_dets if d['color'] == 'yellow'])
        if len(azuis_in) == 0: azuis_in = np.empty((0, 2))
        if len(amarelos_in) == 0: amarelos_in = np.empty((0, 2))

        # Executar cálculo da midline do aluno
        try:
            midline = modulo.calcular_midline_trajetoria(azuis_in, amarelos_in)
            if not isinstance(midline, np.ndarray):
                midline = np.array(midline, dtype=float).reshape(-1, 2) if midline else np.empty((0, 2))
        except Exception as e:
            print(f"[ERRO] {sc['nome']}: Exceção ao executar midline: {e}")
            midline = np.empty((0, 2))

        gt = track_gen.get_ground_truth_track()
        gt_mid = np.array(gt['midline'])

        num_pts = len(midline)
        
        # Calcular desvio de cada waypoint a linha central real da pista
        desvios = []
        if isinstance(midline, np.ndarray) and num_pts > 0 and midline.ndim == 2:
            for pt in midline:
                d = np.min(np.hypot(gt_mid[:, 0] - pt[0], gt_mid[:, 1] - pt[1]))
                desvios.append(d)
        max_desvio = max(desvios) if desvios else 999.0

        # Criterios geometricos rigorosos de pista
        min_pts_necessarios = 4 if i == 3 else 3
        criterio_pontos = (num_pts >= min_pts_necessarios)
        criterio_desvio = (max_desvio <= 1.0) # Desvio nao pode ultrapassar 1 metro da pista

        valido = isinstance(midline, np.ndarray) and midline.ndim == 2 and criterio_pontos and criterio_desvio
        if valido:
            total_aprovados += 1
            status_tag = "[APROVADO]"
            color_title = "#006622"
            sub_title = f"{status_tag} • Waypoints: {num_pts} | Desvio Pista: {max_desvio:.2f}m (OK)"
        else:
            status_tag = "[FALHOU]"
            color_title = "#b30000"
            if sc["drop"] is not None and num_pts == 0:
                sub_title = f"{status_tag} • Curva Cega: Fallback nao implementado (0 waypoints)!"
            elif not criterio_desvio and num_pts > 0:
                sub_title = f"{status_tag} • Desvio Excessivo ({max_desvio:.2f}m > 1.0m)! Emparelhamento errado."
            else:
                sub_title = f"{status_tag} • Cobertura insuficiente: {num_pts} waypoints (minimo {min_pts_necessarios})"

        # ---------------- PLOTAGEM LEVE E CARREGADA ----------------
        mask_mid = (np.abs(gt_mid[:, 0] - car_x) < 30) & (np.abs(gt_mid[:, 1] - car_y) < 30)
        if np.any(mask_mid):
            ax.plot(gt_mid[mask_mid, 0], gt_mid[mask_mid, 1], ':', color='gray', alpha=0.35, label='Traçado Real Pista')

        # Cones de entrada
        if len(azuis_in) > 0:
            ax.plot(azuis_in[:, 0], azuis_in[:, 1], 'o', color='royalblue', markersize=8, markeredgecolor='navy', markeredgewidth=1.5, label=f'Azuis ({len(azuis_in)})')
        if len(amarelos_in) > 0:
            ax.plot(amarelos_in[:, 0], amarelos_in[:, 1], 'o', color='gold', markersize=8, markeredgecolor='darkorange', markeredgewidth=1.5, label=f'Amarelos ({len(amarelos_in)})')

        # SÓ MOSTRA LINHAS DE EMPARELHAMENTO SE O ALUNO TIVER GERADO MIDLINE!
        # E com traço bem carregado/nítido como pedido:
        if valido and len(azuis_in) > 0 and len(amarelos_in) > 0:
            pair_x, pair_y = [], []
            for b in azuis_in:
                dists = np.hypot(amarelos_in[:, 0] - b[0], amarelos_in[:, 1] - b[1])
                closest_idx = np.argmin(dists)
                if dists[closest_idx] <= 6.0:
                    closest_y = amarelos_in[closest_idx]
                    pair_x.extend([b[0], closest_y[0], None])
                    pair_y.extend([b[1], closest_y[1], None])
            if pair_x:
                ax.plot(pair_x, pair_y, color='#00cc44', linestyle='--', linewidth=2.0, alpha=0.85, label='Emparelhamento Validado')

        # MIDLINE DO ALUNO: Bem carregada e destacada a Magenta Viva
        if len(midline) > 0:
            ax.plot(midline[:, 0], midline[:, 1], 'o--', color='#e6007e', linewidth=3.2, 
                    markersize=9, markeredgecolor='black', markeredgewidth=1.5, label=f'Midline Aluno ({len(midline)} pts)')

        # Posição do carro
        ax.plot(car_x, car_y, '^', color='black', markersize=10, markeredgecolor='white', label='Carro')
        ax.arrow(car_x, car_y, 2.5 * np.cos(yaw), 2.5 * np.sin(yaw), head_width=0.8, color='black', alpha=0.8)

        ax.set_xlim(car_x - 12, car_x + 16)
        ax.set_ylim(car_y - 12, car_y + 16)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, linestyle=':', alpha=0.45)

        ax.set_title(f"{sc['nome']}\n{sub_title}", fontsize=9.5, fontweight='bold', color=color_title, pad=8)

        print(f"-> {sc['nome']}:")
        print(f"   Entrada: {len(azuis_in)} Azuis, {len(amarelos_in)} Amarelos")
        print(f"   Waypoints Calculados: {len(midline)} pontos")
        print(f"   Resultado: {status_tag}\n")

    # Legenda global no fundo
    leg_handles = [
        plt.Line2D([0], [0], color='gray', linestyle=':', label='Traçado Real Pista'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', markeredgecolor='navy', markersize=8, label='Cones Azuis'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', markeredgecolor='darkorange', markersize=8, label='Cones Amarelos'),
        plt.Line2D([0], [0], color='#00cc44', linestyle='--', linewidth=2.2, label='Linhas de Emparelhamento'),
        plt.Line2D([0], [0], color='#e6007e', marker='o', linestyle='--', linewidth=3.2, markersize=8, label='Midline Aluno (Trajetória)'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markeredgecolor='black', markersize=8, label='Carro')
    ]
    fig.legend(handles=leg_handles, loc='lower center', ncol=6, fontsize=8.5, frameon=True, 
               framealpha=0.92, edgecolor='#cccccc', bbox_to_anchor=(0.5, 0.02))

    # Margens generosas: ZERO colisão entre títulos e gráficos
    plt.subplots_adjust(top=0.86, bottom=0.08, left=0.06, right=0.95, hspace=0.35, wspace=0.18)

    _ativar_zoom_roda(fig, axes)
    _salvar_ou_mostrar("avaliacao_estatica_etapa2.png", salvar)

    return total_aprovados == len(cenarios)

# =============================================================================
# 3. AVALIAÇÃO ESTÁTICA DEDICADA: ETAPA 3 (CONTROLO / PURE PURSUIT)
# =============================================================================
def avaliar_etapa3(modulo, track_gen: TrackGenerator, salvar=False):
    """
    Avaliação Estática Pura da Tarefa 3 (Controlo):
    Avalia a resposta do algoritmo Pure Pursuit em 4 situações geométricas de condução:
    1. Alinhado na Reta (Erro de rumo nulo -> Steer deve ser aprox 0.0° e acelerar)
    2. Correção de Rumo à Esquerda (Curva à esquerda -> Steer positivo)
    3. Correção de Rumo à Direita (Curva à direita -> Steer negativo)
    4. Saturação Física no Ângulo Máximo (+-0.50 rad / +-28.6°)
    """
    print("\n" + "="*70)
    print(" AVALIAÇÃO ESTÁTICA DEDICADA - ETAPA 3: AS - CONTROLO PURE PURSUIT")
    print("="*70)

    # 4 Casos Estáticos de Controlo
    casos = [
        {
            "nome": "Caso 1: Reta Perfeita (Steer Esperado ~ 0.0° e APPS > 20%)",
            "car": CarState(x=0.0, y=0.0, yaw=0.0, v=8.0, steer=0.0),
            "wps": np.array([[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0]]),
            "check": lambda s, a, b: (abs(s) < 0.05) and (a >= 20.0)
        },
        {
            "nome": "Caso 2: Curva à Esquerda (Steer Esperado > 0.0)",
            "car": CarState(x=0.0, y=0.0, yaw=0.0, v=7.0, steer=0.0),
            "wps": np.array([[2.0, 0.5], [4.0, 1.8], [6.0, 3.5], [8.0, 5.5]]),
            "check": lambda s: s > 0.10
        },
        {
            "nome": "Caso 3: Curva à Direita (Steer Esperado < 0.0)",
            "car": CarState(x=0.0, y=0.0, yaw=0.0, v=7.0, steer=0.0),
            "wps": np.array([[2.0, -0.5], [4.0, -1.8], [6.0, -3.5], [8.0, -5.5]]),
            "check": lambda s: s < -0.10
        },
        {
            "nome": "Caso 4: Saturação Física do Volante (Máx ±0.50 rad)",
            "car": CarState(x=0.0, y=0.0, yaw=0.0, v=5.0, steer=0.0),
            "wps": np.array([[1.0, 4.0], [2.0, 8.0], [3.0, 12.0]]), # Alvo com elevado desvio lateral
            "check": lambda s: abs(abs(s) - 0.50) < 0.02
        },
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("ETAPA 3 [AS - CONTROLO]: Avaliação Estática do Algoritmo Pure Pursuit\n(Vetores de Mira Lookahead, Curvatura de Viragem e Saturação Física a ±0.50 rad)", 
                 fontsize=12.5, fontweight='bold', y=0.965)
    axes = axes.flatten()

    total_aprovados = 0

    for i, cs in enumerate(casos):
        ax = axes[i]
        car = cs["car"]
        wps = cs["wps"]

        excecao_msg = None
        try:
            res = modulo.controlador_pure_pursuit(car, wps, lookahead_distance=4.0, wheelbase=1.55)
            if isinstance(res, (tuple, list)) and len(res) >= 3:
                steer, apps, brake = float(res[0]), float(res[1]), float(res[2])
            elif isinstance(res, (float, np.floating, int)):
                steer, apps, brake = float(res), 50.0, 0.0
            else:
                steer, apps, brake = 0.0, 0.0, 0.0
        except Exception as e:
            excecao_msg = str(e)
            print(f"[ERRO] {cs['nome']}: Exceção ao executar Pure Pursuit: {e}")
            steer, apps, brake = 0.0, 0.0, 0.0

        if excecao_msg is not None:
            aprovado = False
            status_tag = "[FALHOU]"
            sub_title = f"[FALHOU] (Exceção: {excecao_msg})"
            color_title = "darkred"
        else:
            if cs["nome"].startswith("Caso 1"):
                aprovado = cs["check"](steer, apps, brake)
            else:
                steer_ok = cs["check"](steer)
                pedais_ok = (0.0 <= apps <= 100.0) and (0.0 <= brake <= 80.0)
                aprovado = steer_ok and pedais_ok

            if aprovado:
                total_aprovados += 1
                status_tag = "[APROVADO]"
                color_title = "darkgreen"
            else:
                status_tag = "[FALHOU]"
                color_title = "darkred"
            sub_title = f"{status_tag} Volante: {steer:+.3f} rad ({np.degrees(steer):+.1f}°) | APPS: {apps:.1f}% | Travão: {brake:.1f} bar"

        # 1. Círculo de Mira Lookahead (Raio = 4.0m)
        lookahead_circle = plt.Circle((car.x, car.y), 4.0, color='#888888', linestyle=':', fill=False, linewidth=1.5, alpha=0.7)
        ax.add_patch(lookahead_circle)

        # 2. Trajetória Central (Midline)
        ax.plot(wps[:, 0], wps[:, 1], 'o--', color='#e6007e', linewidth=2.5, markersize=7)

        # 3. Ponto Alvo Selecionado (Target)
        dists = np.hypot(wps[:, 0] - car.x, wps[:, 1] - car.y)
        idx_target = np.argmin(np.abs(dists - 4.0))
        tx, ty = wps[idx_target]
        ax.plot(tx, ty, 'o', color='#ff007f', markersize=11, markeredgecolor='black', markeredgewidth=1.8)
        ax.plot([car.x, tx], [car.y, ty], color='black', linestyle='--', linewidth=1.3, alpha=0.55)

        # 4. Vetor de Rumo Atual do Veículo
        ax.arrow(car.x, car.y, 2.8 * np.cos(car.yaw), 2.8 * np.sin(car.yaw), 
                 head_width=0.4, color='black', alpha=0.75, linewidth=1.2)

        # 5. Vetor de Viragem Comandado (apenas se executou sem exceção)
        if excecao_msg is None:
            steer_yaw = car.yaw + steer
            arrow_color = '#00cc44' if aprovado else '#cc0000'
            ax.arrow(car.x, car.y, 3.2 * np.cos(steer_yaw), 3.2 * np.sin(steer_yaw), 
                     head_width=0.45, color=arrow_color, linewidth=2.4)
        else:
            # Sem resposta / não implementado
            ax.text(car.x + 0.5, car.y - 1.2, "[SEM COMANDO / EXCEÇÃO]", color='darkred', 
                    fontweight='bold', fontsize=8, bbox=dict(boxstyle='square,pad=0.2', facecolor='#ffe6e6', edgecolor='red', alpha=0.85))

        # 6. Monolugar (Posição e Orientação)
        ax.plot(car.x, car.y, '^', color='black', markersize=11, markeredgecolor='white', markeredgewidth=1.5)

        # 7. Caixa de Comandos dos Atuadores
        pedal_box = f"Comandos Atuadores:\n• Volante : {np.degrees(steer):+.1f}° ({steer:+.3f} rad)\n• APPS    : {apps:.1f} %\n• Travão  : {brake:.1f} bar"
        box_y = 0.96 if cs["nome"].startswith("Caso 3") else 0.96
        ax.text(0.03, box_y, pedal_box, transform=ax.transAxes, verticalalignment='top',
                fontsize=8.5, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#999999', alpha=0.92))

        # 8. Limites Dinâmicos Inteligentes (evita qualquer corte de waypoints)
        all_x = np.concatenate([[car.x - 4.5, car.x + 4.5], wps[:, 0]])
        all_y = np.concatenate([[car.y - 4.5, car.y + 4.5], wps[:, 1]])
        margin = 1.8
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, linestyle=':', alpha=0.45)

        ax.set_title(f"{cs['nome']}\n{sub_title}", 
                     fontsize=9.2, fontweight='bold', color=color_title, pad=9)

        print(f"-> {cs['nome']}:")
        print(f"   Ângulo de Volante Retornado: {steer:.4f} rad ({np.degrees(steer):.2f}°)")
        print(f"   Comandos Longitudinais     : APPS {apps:.1f}% | Travão {brake:.1f} bar")
        print(f"   Critério Aprovado          : {status_tag}\n")

    # Legenda global no rodapé (mostra ambos os estados: verde se aprovado, vermelho se falhou)
    leg_handles = [
        plt.Line2D([0], [0], color='#e6007e', marker='o', linestyle='--', linewidth=2.5, markersize=7, label='Midline Alvo'),
        plt.Line2D([0], [0], color='#ff007f', marker='o', markeredgecolor='black', markersize=9, linestyle='None', label='Ponto de Mira Alvo'),
        plt.Line2D([0], [0], color='#888888', linestyle=':', linewidth=1.5, label='Círculo Lookahead (4.0m)'),
        plt.Line2D([0], [0], color='black', linestyle='-', linewidth=1.5, label='Rumo do Monolugar'),
        plt.Line2D([0], [0], color='#00cc44', linestyle='-', linewidth=2.8, label='Volante (Aprovado)'),
        plt.Line2D([0], [0], color='#cc0000', linestyle='-', linewidth=2.8, label='Volante (Incorreto)'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markeredgecolor='black', markersize=9, label='Monolugar')
    ]
    fig.legend(handles=leg_handles, loc='lower center', ncol=7, fontsize=8.2, frameon=True, 
               framealpha=0.92, edgecolor='#cccccc', bbox_to_anchor=(0.5, 0.02))

    plt.subplots_adjust(top=0.86, bottom=0.08, left=0.06, right=0.95, hspace=0.35, wspace=0.18)
    _ativar_zoom_roda(fig, axes)
    _salvar_ou_mostrar("avaliacao_estatica_etapa3.png", salvar)

    return total_aprovados == len(casos)


# =============================================================================
# 4. AVALIAÇÃO ESTÁTICA DEDICADA: ETAPA 4 (ESW - TELEMETRIA & SLIP)
# =============================================================================
def avaliar_etapa4(modulo, salvar=False):
    """
    Avaliação Estática Pura da Tarefa 4 (Telemetria):
    Gera uma sequência temporal CAN sincronizada (9.5 segundos a 30 Hz) contendo:
    1. Partida suave e proteção de baixa velocidade (< 1.0 m/s)
    2. Teste de Limiar Spin (0.13 < 0.15 -> OFF) seguido de Spin Real (+0.35 > 0.15 -> ATIVO)
    3. Reta estável com ruído pontual de 1 amostra de encoder (teste de rejeição de falso positivo)
    4. Teste de Limiar Lock (-0.14 > -0.20 -> OFF) seguido de Lock Sustentado (-0.65 < -0.20 -> ATIVO)
    5. Teste de Limiar Carga Lateral (1.30G < 1.40G -> OFF) seguido de Curva Fechada (1.60G > 1.40G -> ATIVO)
    6. Imobilização total
    """
    print("\n" + "="*70)
    print(" AVALIAÇÃO ESTÁTICA DEDICADA - ETAPA 4: ESW - TELEMETRIA & ALARMES CAN")
    print("="*70)

    # 1. Gerar traço de telemetria CAN sincronizado (9.5s a ~30Hz)
    dt = 0.035
    time_vec = np.arange(0.0, 9.5, dt)
    N = len(time_vec)

    v_car = np.zeros(N)
    v_fl = np.zeros(N)
    v_fr = np.zeros(N)
    v_rl = np.zeros(N)
    v_rr = np.zeros(N)
    apps = np.zeros(N)
    brake = np.zeros(N)
    ay = np.zeros(N)

    gt_spin = np.zeros(N, dtype=bool)
    gt_lock = np.zeros(N, dtype=bool)
    gt_glitch = np.zeros(N, dtype=bool)
    gt_high_g = np.zeros(N, dtype=bool)

    for i, t in enumerate(time_vec):
        if t < 0.8:
            v_car[i] = 0.3 + 0.6 * (t / 0.8)  # < 1.0 m/s (teste de divisão por zero)
            v_fl[i], v_fr[i], v_rl[i], v_rr[i] = v_car[i], v_car[i], v_car[i], v_car[i]
            apps[i] = 20.0
        elif 0.8 <= t < 1.8:
            # Spin Sub-Limiar: acelera a fundo mas slip = +0.13 (< 0.15 -> SEM alarme)
            v_car[i] = 1.0 + 3.0 * (t - 0.8)
            apps[i] = 75.0
            v_rl[i] = v_car[i] * 1.13
            v_rr[i] = v_car[i] * 1.12
            v_fl[i], v_fr[i] = v_car[i], v_car[i]
        elif 1.8 <= t < 3.2:
            # Spin Real Acima do Limiar: slip = +0.35 (> 0.15 -> COM alarme)
            v_car[i] = 4.0 + 3.5 * (t - 1.8)
            apps[i] = 85.0
            gt_spin[i] = True
            v_rl[i] = v_car[i] * 1.35
            v_rr[i] = v_car[i] * 1.33
            v_fl[i], v_fr[i] = v_car[i], v_car[i]
        elif 3.2 <= t < 4.2:
            # Reta estável + ruído pontual de 1 amostra de encoder a t=3.7s
            v_car[i] = 8.9 + 1.2 * (t - 3.2)
            apps[i] = 30.0
            v_rl[i], v_rr[i], v_fl[i], v_fr[i] = v_car[i], v_car[i], v_car[i], v_car[i]
            if abs(t - 3.7) < dt / 2.0:
                v_fl[i] = v_car[i] * 0.20  # Pico isolado de bloqueio
                brake[i] = 50.0
                gt_glitch[i] = True
        elif 4.2 <= t < 5.0:
            # Lock Sub-Limiar: travagem com escorregamento moderado (-0.14 > -0.20 -> SEM alarme)
            v_car[i] = 10.1 - 2.5 * (t - 4.2)
            brake[i] = 40.0
            v_fl[i] = v_car[i] * 0.86
            v_fr[i] = v_car[i] * 0.88
            v_rl[i], v_rr[i] = v_car[i], v_car[i]
        elif 5.0 <= t < 5.8:
            # Lock Real Sustentado: rodas dianteiras bloqueadas (-0.65 < -0.20 -> COM alarme à 3ª amostra)
            v_car[i] = 8.1 - 4.5 * (t - 5.0)
            brake[i] = 65.0
            gt_lock[i] = True
            v_fl[i] = v_car[i] * 0.35
            v_fr[i] = v_car[i] * 0.38
            v_rl[i], v_rr[i] = v_car[i], v_car[i]
        elif 5.8 <= t < 6.8:
            # Curva Sub-Limiar: aceleração lateral moderada (1.30G < 1.40G -> SEM alarme)
            v_car[i] = 8.0
            apps[i] = 35.0
            ay[i] = 12.75  # 12.75 / 9.81 = 1.30 G
            v_fl[i], v_fr[i], v_rl[i], v_rr[i] = v_car[i], v_car[i], v_car[i], v_car[i]
        elif 6.8 <= t < 8.2:
            # Curva Fechada Alta Carga: aceleração lateral extrema (1.60G > 1.40G -> COM alarme)
            v_car[i] = 8.0
            apps[i] = 45.0
            ay[i] = 15.70  # 15.70 / 9.81 = 1.60 G
            gt_high_g[i] = True
            v_fl[i], v_fr[i], v_rl[i], v_rr[i] = v_car[i], v_car[i], v_car[i], v_car[i]
        else:
            # Imobilização
            v_car[i] = max(0.0, 8.0 - 6.0 * (t - 8.2))
            brake[i] = 30.0
            v_fl[i], v_fr[i], v_rl[i], v_rr[i] = v_car[i], v_car[i], v_car[i], v_car[i]

    # 2. Executar algoritmo do aluno / referência em cada amostra CAN
    aluno_spin = np.zeros(N, dtype=bool)
    aluno_lock = np.zeros(N, dtype=bool)
    aluno_high_g = np.zeros(N, dtype=bool)
    aluno_rl_slip = np.zeros(N)
    aluno_fl_slip = np.zeros(N)
    excecao_msg = None

    if hasattr(modulo, '_wheel_lock_consecutive_count'):
        modulo._wheel_lock_consecutive_count = 0

    for i in range(N):
        pkt = {
            'est_car_velocity': v_car[i],
            'fl_wheel_speed': v_fl[i],
            'fr_wheel_speed': v_fr[i],
            'rl_wheel_speed': v_rl[i],
            'rr_wheel_speed': v_rr[i],
            'accel_x': 0.0,
            'accel_y': ay[i],
            'apps': apps[i],
            'brake_pressure': brake[i]
        }
        try:
            res = modulo.analisar_dinamica_veiculo(pkt)
            aluno_rl_slip[i] = res.get('rl_slip', 0.0)
            aluno_fl_slip[i] = (v_fl[i] - v_car[i]) / v_car[i] if v_car[i] > 1.0 else 0.0
            avisos = res.get('avisos', [])
            aluno_spin[i] = "AVISO_WHEEL_SPIN" in avisos or res.get('wheel_spin_detected', False)
            aluno_lock[i] = "AVISO_WHEEL_LOCK" in avisos or res.get('wheel_lock_detected', False)
            aluno_high_g[i] = "AVISO_HIGH_LATERAL_G" in avisos
        except Exception as e:
            if excecao_msg is None:
                excecao_msg = str(e)

    # 3. Avaliação rigorosa dos 4 critérios com validação de limiares
    if excecao_msg is not None:
        print(f"[ERRO] Exceção ao executar telemetria: {excecao_msg}")
        spin_aprovado = False
        lock_aprovado = False
        glitch_rejeitado = False
        high_g_aprovado = False
    else:
        # Spin: ativa no spin real e NÃO ativa na fase sub-limiar (0.8s <= t < 1.8s) nem em reta estável
        sub_spin = (time_vec >= 0.8) & (time_vec < 1.8)
        spin_aprovado = np.any(aluno_spin & gt_spin) and not np.any(aluno_spin & sub_spin)

        # Lock: ativa no lock sustentado e NÃO ativa no sub-limiar (4.2s <= t < 5.0s)
        sub_lock = (time_vec >= 4.2) & (time_vec < 5.0)
        lock_aprovado = np.any(aluno_lock & gt_lock) and not np.any(aluno_lock & sub_lock)

        # Glitch: rejeita o pico de 1 amostra isolada
        glitch_rejeitado = not np.any(aluno_lock & gt_glitch)

        # High G: ativa na curva apertada e NÃO ativa no sub-limiar (5.8s <= t < 6.8s)
        sub_high_g = (time_vec >= 5.8) & (time_vec < 6.8)
        high_g_aprovado = np.any(aluno_high_g & gt_high_g) and not np.any(aluno_high_g & sub_high_g)

    total_aprovados = sum([spin_aprovado, lock_aprovado, glitch_rejeitado, high_g_aprovado])
    tudo_aprovado = (total_aprovados == 4) and (excecao_msg is None)

    print(f"-> Deteção de Wheel Spin (Limiar +0.15): {'[APROVADO]' if spin_aprovado else '[FALHOU]'}")
    print(f"-> Rejeição de Ruído Isolado (Debounce 3 Amostras): {'[APROVADO]' if glitch_rejeitado else '[FALHOU]'}")
    print(f"-> Deteção de Wheel Lock (Limiar -0.20): {'[APROVADO]' if lock_aprovado else '[FALHOU]'}")
    print(f"-> Deteção de Alta Carga Lateral (Limiar 1.4G): {'[APROVADO]' if high_g_aprovado else '[FALHOU]'}\n")

    # 4. Construção dos 3 Subplots Sincronizados
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15.5, 10.8), sharex=True)
    fig.suptitle("ETAPA 4 [ESW - TELEMETRIA]: Análise Temporal de Dinâmica Veicular & Alarmes CAN", 
                 fontsize=13, fontweight='bold', y=0.978)

    # Placar de Resumo no Topo (caixa destacada, sem colisão com eixos ou rótulos)
    resumo_color = '#15803d' if tudo_aprovado else '#b91c1c'
    resumo_bg = '#f0fdf4' if tudo_aprovado else '#fef2f2'
    resumo_border = '#86efac' if tudo_aprovado else '#fca5a5'
    resumo_texto = (
        f"Resultado da Avaliação: {total_aprovados}/4 Critérios Aprovados   |   "
        f"Wheel Spin (Limiar +0.15): {'OK' if spin_aprovado else 'FAIL'}   •   "
        f"Ruído Rejeitado (Debounce): {'OK' if glitch_rejeitado else 'FAIL'}   •   "
        f"Wheel Lock (Limiar -0.20): {'OK' if lock_aprovado else 'FAIL'}   •   "
        f"Carga Lateral (Limiar 1.4G): {'OK' if high_g_aprovado else 'FAIL'}"
    )
    fig.text(0.5, 0.942, resumo_texto, ha='center', va='center', fontsize=9.5, fontweight='bold', 
             color=resumo_color, bbox=dict(boxstyle='round,pad=0.38', facecolor=resumo_bg, edgecolor=resumo_border, linewidth=1.2))

    # --- SUBPLOT 1: Sinais CAN de Entrada (Velocidades & Pedais) ---
    ax1.set_title("1. Sinais CAN: Velocidades de Roda vs Carro e Pedais", fontsize=9.5, fontweight='bold', loc='left', pad=10)
    l_vcar, = ax1.plot(time_vec, v_car, color='black', linewidth=2.2, label='V_Carro (m/s)')
    l_vrl, = ax1.plot(time_vec, v_rl, color='#d97706', linestyle='--', linewidth=1.6, label='V_Roda RL (Traseira)')
    l_vfl, = ax1.plot(time_vec, v_fl, color='#dc2626', linestyle=':', linewidth=1.6, label='V_Roda FL (Dianteira)')
    ax1.set_ylabel("Velocidade (m/s)", fontsize=9, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.5)

    ax1_pedals = ax1.twinx()
    l_apps, = ax1_pedals.plot(time_vec, apps, color='#16a34a', alpha=0.45, linewidth=1.4, label='APPS (%)')
    l_brake, = ax1_pedals.plot(time_vec, brake, color='#ef4444', alpha=0.45, linewidth=1.4, label='Brake (bar)')
    ax1_pedals.set_ylabel("Pedais (APPS % / Brake bar)", fontsize=8.5, color='#475569')
    ax1_pedals.set_ylim(-5, 105)

    ax1.legend(handles=[l_vcar, l_vrl, l_vfl, l_apps, l_brake], loc='lower right', bbox_to_anchor=(1.0, 1.01),
               ncol=5, fontsize=8, frameon=False)

    # --- SUBPLOT 2: Slip Ratios & Limiares ---
    ax2.set_title("2. Escorregamento Longitudinal (Slip Ratio) vs Limiares de Disparo", fontsize=9.5, fontweight='bold', loc='left', pad=10)
    l_slip_rl, = ax2.plot(time_vec, aluno_rl_slip, color='#d97706', linewidth=2.0, label='Slip Traseiro RL')
    l_slip_fl, = ax2.plot(time_vec, aluno_fl_slip, color='#dc2626', linewidth=1.8, label='Slip Dianteiro FL')
    l_lim_spin = ax2.axhline(0.15, color='#d97706', linestyle='--', linewidth=1.3, alpha=0.85, label='Limiar Spin (+0.15)')
    l_lim_lock = ax2.axhline(-0.20, color='#dc2626', linestyle='--', linewidth=1.3, alpha=0.85, label='Limiar Lock (-0.20)')
    ax2.axhline(0.0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.set_ylabel("Slip Ratio", fontsize=9, fontweight='bold')
    ax2.set_ylim(-0.85, 0.60)
    ax2.grid(True, linestyle=':', alpha=0.5)

    # Destaques visuais das fases de limiar
    ax2.text(1.3, 0.22, "Sub-Limiar\n(0.13 < 0.15)", color='#b45309', fontsize=7.5, ha='center', fontweight='bold', alpha=0.85)
    ax2.text(2.5, 0.42, "Acima Limiar\n(+0.35)", color='#b45309', fontsize=7.5, ha='center', fontweight='bold', alpha=0.85)
    ax2.text(4.6, -0.32, "Sub-Limiar\n(-0.14 > -0.20)", color='#991b1b', fontsize=7.5, ha='center', fontweight='bold', alpha=0.85)
    ax2.text(5.4, -0.76, "Acima Limiar\n(-0.65)", color='#991b1b', fontsize=7.5, ha='center', fontweight='bold', alpha=0.85)

    ax2.legend(handles=[l_slip_rl, l_slip_fl, l_lim_spin, l_lim_lock], loc='lower right', bbox_to_anchor=(1.0, 1.01),
               ncol=4, fontsize=8, frameon=False)

    # --- SUBPLOT 3: Canais Digitais de Alarme (Logic Analyzer CAN) ---
    ax3.set_title("3. Canais Digitais de Alarme CAN (Estilo Logic Analyzer)", 
                  fontsize=9.5, fontweight='bold', loc='left', pad=10)
    ax3.text(1.0, 1.03, "Sombreado = Evento Real  |  Linha = Código Aluno  |  Pontilhado = Ruído Rejeitado",
             transform=ax3.transAxes, ha='right', fontsize=8, color='#475569', fontstyle='italic')

    # Faixa 1 (Inferior): AVISO_WHEEL_SPIN
    ax3.axhspan(-0.15, 0.85, color='#fefce8', alpha=0.45)
    ax3.axvspan(1.8, 3.2, ymin=0.0/3.0, ymax=1.0/3.0, color='#fde047', alpha=0.55)
    spin_line_color = '#ca8a04' if spin_aprovado else '#dc2626'
    ax3.text(1.3, 0.40, "[Sub-Limiar: OFF]", color='#854d0e', fontsize=7.5, ha='center', fontweight='bold', alpha=0.8)

    # Faixa 2 (Intermédia): AVISO_WHEEL_LOCK
    ax3.axhspan(0.85, 1.85, color='#fef2f2', alpha=0.45)
    ax3.axvspan(5.0, 5.8, ymin=1.0/3.0, ymax=2.0/3.0, color='#fca5a5', alpha=0.55)
    ax3.axvline(3.7, ymin=1.0/3.0, ymax=2.0/3.0, color='#991b1b', linestyle=':', linewidth=2.2)
    lock_line_color = '#dc2626' if (lock_aprovado and glitch_rejeitado) else '#991b1b'
    ax3.text(4.6, 1.40, "[Sub-Limiar: OFF]", color='#991b1b', fontsize=7.5, ha='center', fontweight='bold', alpha=0.8)

    # Faixa 3 (Superior): AVISO_HIGH_LATERAL_G
    ax3.axhspan(1.85, 2.85, color='#faf5ff', alpha=0.45)
    ax3.axvspan(6.8, 8.2, ymin=2.0/3.0, ymax=3.0/3.0, color='#e9d5ff', alpha=0.55)
    high_g_line_color = '#7c3aed' if high_g_aprovado else '#dc2626'
    ax3.text(6.3, 2.40, "[Sub-Limiar: OFF]", color='#6b21a8', fontsize=7.5, ha='center', fontweight='bold', alpha=0.8)

    if excecao_msg is not None:
        ax3.text(4.75, 1.35, f"[FALHOU: Exceção ao Executar - {excecao_msg}]", 
                 color='darkred', ha='center', va='center', fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#fee2e2', edgecolor='red', alpha=0.9))
    else:
        ax3.step(time_vec, aluno_spin.astype(int) * 0.65 + 0.05, where='post', color=spin_line_color, linewidth=2.2)
        ax3.step(time_vec, aluno_lock.astype(int) * 0.65 + 1.05, where='post', color=lock_line_color, linewidth=2.2)
        ax3.step(time_vec, aluno_high_g.astype(int) * 0.65 + 2.05, where='post', color=high_g_line_color, linewidth=2.2)

    ax3.set_yticks([0.05, 0.70, 1.05, 1.70, 2.05, 2.70])
    ax3.set_yticklabels(['OFF', 'ATIVO', 'OFF', 'ATIVO', 'OFF', 'ATIVO'], fontsize=7.5)

    ax3.text(-0.015, 0.375, "CANAL 1:\nWHEEL_SPIN", transform=ax3.get_yaxis_transform(), 
             ha='right', va='center', fontsize=8, fontweight='bold', color='#ca8a04')
    ax3.text(-0.015, 1.375, "CANAL 2:\nWHEEL_LOCK", transform=ax3.get_yaxis_transform(), 
             ha='right', va='center', fontsize=8, fontweight='bold', color='#dc2626')
    ax3.text(-0.015, 2.375, "CANAL 3:\nHIGH_LATERAL_G", transform=ax3.get_yaxis_transform(), 
             ha='right', va='center', fontsize=8, fontweight='bold', color='#7c3aed')

    ax3.set_xlabel("Tempo de Telemetria CAN (segundos)", fontsize=9.5, fontweight='bold')
    ax3.set_xlim(0, 9.5)
    ax3.set_ylim(-0.15, 2.85)
    ax3.grid(True, linestyle=':', alpha=0.5)

    plt.subplots_adjust(top=0.89, bottom=0.07, left=0.12, right=0.94, hspace=0.40)
    _ativar_zoom_roda(fig, [ax1, ax2, ax3])
    _salvar_ou_mostrar("avaliacao_estatica_etapa4.png", salvar)

    return tudo_aprovado


# =============================================================================
# 5. AVALIAÇÃO ESTÁTICA DEDICADA: ETAPA 5 (ESW - SAFETY STATE MACHINE & EBS)
# =============================================================================
def avaliar_etapa5(modulo, salvar=False):
    """
    Avaliação Estática Pura da Tarefa 5 (Safety & EBS):
    Testa rigorosamente a máquina de estados e gatilhos de segurança FSG:
    1. Permanência em AS_READY sem comando de largada
    2. Transição AS_READY -> AS_DRIVING com comando de largada (Start Cmd = SIM)
    3. Condução Nominal em pista
    4. Teste de Limiar de Pista (Sub-Limiar d=1.8m <= 2.0m -> Sem EBS)
    5. Gatilho de Emergência: Saída de Pista (Off-Track d=2.3m > 2.0m -> EBS Ativado)
    6. Gatilho de Emergência: Perda Total de Cones (0 cones visíveis -> EBS Ativado)
    7. Trava de Bloqueio em Emergência (irrecuperável na pista)
    """
    print("\n" + "="*70)
    print(" AVALIAÇÃO ESTÁTICA DEDICADA - ETAPA 5: ESW - SAFETY STATE MACHINE & EBS")
    print("="*70)

    test_cases = [
        {"desc": "1. Espera em Grelha (Sem Comando)", "init": "AS_READY", "d": 0.0, "c": 10, "cmd": False, "exp_state": "AS_READY", "exp_ebs": False},
        {"desc": "2. Comando de Largada da Missão", "init": "AS_READY", "d": 0.0, "c": 10, "cmd": True, "exp_state": "AS_DRIVING", "exp_ebs": False},
        {"desc": "3. Condução Nominal em Pista", "init": "AS_DRIVING", "d": 0.4, "c": 12, "cmd": False, "exp_state": "AS_DRIVING", "exp_ebs": False},
        {"desc": "4. Limiar de Pista (Sub-Limiar <= 2.0m)", "init": "AS_DRIVING", "d": 1.8, "c": 10, "cmd": False, "exp_state": "AS_DRIVING", "exp_ebs": False},
        {"desc": "5. Saída de Pista (Off-Track > 2.0m)", "init": "AS_DRIVING", "d": 2.3, "c": 8, "cmd": False, "exp_state": "AS_EMERGENCY", "exp_ebs": True},
        {"desc": "6. Perda Total de Visão (0 Cones)", "init": "AS_DRIVING", "d": 0.3, "c": 0, "cmd": False, "exp_state": "AS_EMERGENCY", "exp_ebs": True},
        {"desc": "7. Trava Bloqueada em Emergência", "init": "AS_EMERGENCY", "d": 0.0, "c": 15, "cmd": True, "exp_state": "AS_EMERGENCY", "exp_ebs": True},
    ]

    total_aprovados = 0
    table_rows = []

    for tc in test_cases:
        try:
            n_state, ebs = modulo.safety_state_machine_ebs(tc["init"], tc["d"], tc["c"], tc["cmd"])
            aprovado = (n_state == tc["exp_state"] and ebs == tc["exp_ebs"])
            if aprovado:
                total_aprovados += 1
            status_tag = "APROVADO" if aprovado else "FALHOU"
            ebs_str = "ATIVADO (True)" if ebs else "DESLIGADO (False)"
        except NotImplementedError:
            n_state = "---"
            ebs_str = "---"
            status_tag = "FALHOU (Por implementar)"
            aprovado = False
        except Exception as e:
            n_state = "ERRO"
            ebs_str = "ERRO"
            status_tag = f"FALHOU ({str(e)[:25]})"
            aprovado = False

        table_rows.append({
            "data": [
                tc["desc"], 
                tc["init"], 
                f"{tc['d']:.1f} m", 
                str(tc["c"]), 
                "SIM" if tc["cmd"] else "NÃO", 
                n_state, 
                ebs_str, 
                status_tag
            ],
            "aprovado": aprovado
        })

        print(f"-> {tc['desc']}:")
        print(f"   Transição: {tc['init']} -> {n_state} | EBS: {ebs_str} -> [{status_tag}]\n")

    tudo_aprovado = (total_aprovados == len(test_cases))

    # Criação da Figura com 2 Painéis
    fig = plt.figure(figsize=(15.5, 9.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.95, 1.85], hspace=0.18)

    # 1. Cabeçalho Geral
    fig.suptitle("ETAPA 5 [ESW - SAFETY]: Validação da Máquina de Estados & Gatilhos EBS", 
                 fontsize=13, fontweight='bold', y=0.975)

    resumo_color = '#15803d' if tudo_aprovado else '#b91c1c'
    resumo_bg = '#f0fdf4' if tudo_aprovado else '#fef2f2'
    resumo_border = '#86efac' if tudo_aprovado else '#fca5a5'
    resumo_texto = (
        f"Resultado da Avaliação: {total_aprovados}/{len(test_cases)} Casos de Segurança Aprovados   |   "
        f"Grelha/Largada: {'OK' if (table_rows[0]['aprovado'] and table_rows[1]['aprovado']) else 'FAIL'}   •   "
        f"Condução Nominal: {'OK' if table_rows[2]['aprovado'] else 'FAIL'}   •   "
        f"Limiar Pista: {'OK' if table_rows[3]['aprovado'] else 'FAIL'}   •   "
        f"Gatilho Off-Track: {'OK' if table_rows[4]['aprovado'] else 'FAIL'}   •   "
        f"Perda Cones: {'OK' if table_rows[5]['aprovado'] else 'FAIL'}   •   "
        f"Trava EBS: {'OK' if table_rows[6]['aprovado'] else 'FAIL'}"
    )
    fig.text(0.5, 0.938, resumo_texto, ha='center', va='center', fontsize=9.2, fontweight='bold', 
             color=resumo_color, bbox=dict(boxstyle='round,pad=0.38', facecolor=resumo_bg, edgecolor=resumo_border, linewidth=1.2))

    # -------------------------------------------------------------------------
    # PAINEL 1: Diagrama Gráfico da Máquina de Estados Regulamentar FSG
    # -------------------------------------------------------------------------
    ax_diag = fig.add_subplot(gs[0])
    ax_diag.set_title("1. Arquitetura da Máquina de Estados Driverless (Regras Formula Student FSG)", 
                      fontsize=9.8, fontweight='bold', loc='left', pad=8)
    ax_diag.axis('off')
    ax_diag.set_xlim(-0.5, 11.8)
    ax_diag.set_ylim(-0.2, 3.0)

    # 4 Nós de Estado
    states = [
        {"name": "AS_OFF", "x": 0.8, "y": 1.2, "bg": "#f1f5f9", "border": "#64748b", "sub": "Sistema Desligado\n(Repouso Total)"},
        {"name": "AS_READY", "x": 3.7, "y": 1.2, "bg": "#eff6ff", "border": "#2563eb", "sub": "Pronto na Grelha\n(Espera comando)"},
        {"name": "AS_DRIVING", "x": 6.8, "y": 1.2, "bg": "#f0fdf4", "border": "#16a34a", "sub": "Andamento Autónomo\n(EBS Desligado)"},
        {"name": "AS_EMERGENCY", "x": 9.8, "y": 1.2, "bg": "#fef2f2", "border": "#dc2626", "sub": "EBS Disparado!\n(Travão Máximo 40 bar)"},
    ]

    for st in states:
        box = FancyBboxPatch((st["x"] - 0.95, st["y"] - 0.55), 1.9, 1.1,
                             boxstyle="round,pad=0.15", facecolor=st["bg"],
                             edgecolor=st["border"], linewidth=2.0)
        ax_diag.add_patch(box)
        ax_diag.text(st["x"], st["y"] + 0.18, st["name"], ha="center", va="center",
                     fontsize=9.5, fontweight="bold", color=st["border"])
        ax_diag.text(st["x"], st["y"] - 0.22, st["sub"], ha="center", va="center",
                     fontsize=7.2, color="#475569")

    # Setas de Transição
    # Transição 1: AS_OFF -> AS_READY
    arrow1 = FancyArrowPatch((1.8, 1.2), (2.7, 1.2), arrowstyle="->,head_width=4,head_length=6",
                             color="#64748b", linewidth=1.8)
    ax_diag.add_patch(arrow1)
    ax_diag.text(2.25, 1.45, "Armar AS", ha="center", fontsize=7.5, fontweight="bold", color="#475569")

    # Transição 2: AS_READY -> AS_DRIVING (Largada)
    arrow2 = FancyArrowPatch((4.7, 1.2), (5.8, 1.2), arrowstyle="->,head_width=4,head_length=6",
                             color="#16a34a", linewidth=2.0)
    ax_diag.add_patch(arrow2)
    ax_diag.text(5.25, 1.45, "Start Cmd = SIM", ha="center", fontsize=7.8, fontweight="bold", color="#15803d")

    # Transição 3: AS_DRIVING -> AS_EMERGENCY (Gatilhos Críticos)
    arrow3 = FancyArrowPatch((7.8, 1.2), (8.8, 1.2), arrowstyle="->,head_width=4,head_length=6",
                             color="#dc2626", linewidth=2.2)
    ax_diag.add_patch(arrow3)
    ax_diag.text(8.3, 1.62, "GATILHOS EBS:\n1. Off-Track > 2.0m\n2. Cones Vistos == 0", 
                 ha="center", fontsize=7.2, fontweight="bold", color="#b91c1c",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff1f2", edgecolor="#fca5a5", linewidth=0.8))

    # Transição 4: Trava Irrecuperável (Loop em AS_EMERGENCY)
    arrow4 = FancyArrowPatch((10.75, 1.5), (10.75, 0.9), arrowstyle="->,head_width=4,head_length=6",
                             color="#dc2626", linewidth=1.8,
                             connectionstyle="arc3,rad=-1.2")
    ax_diag.add_patch(arrow4)
    ax_diag.text(10.95, 1.2, "Trava\nEBS", 
                 ha="left", va="center", fontsize=7.2, fontweight="bold", color="#991b1b")

    ax_diag.text(9.8, 0.25, "(Irrecuperável na pista: permanece sempre em AS_EMERGENCY)", 
                 ha="center", fontsize=7.2, fontstyle="italic", color="#991b1b")

    # -------------------------------------------------------------------------
    # PAINEL 2: Tabela Formatada Profissional (Zero Texto Cortado)
    # -------------------------------------------------------------------------
    ax_tab = fig.add_subplot(gs[1])
    ax_tab.set_title("2. Tabela de Verificação dos Cenários de Teste", 
                     fontsize=9.8, fontweight='bold', loc='left', pad=8)
    ax_tab.axis('off')

    col_headers = [
        "Cenário de Teste", "Estado Inicial", "Dist. Centro", 
        "Cones", "Start Cmd", "Estado Obtido", "EBS Ativo?", "Avaliação"
    ]
    col_widths = [0.26, 0.11, 0.09, 0.07, 0.09, 0.12, 0.11, 0.15]

    cell_data = [[h for h in col_headers]]
    cell_colors = [["#1e293b"] * len(col_headers)]

    for i, r in enumerate(table_rows):
        cell_data.append(r["data"])
        row_bg = "#ffffff" if i % 2 == 0 else "#f8fafc"
        row_c = [row_bg] * len(col_headers)
        if r["aprovado"]:
            row_c[-1] = "#dcfce7"
        else:
            row_c[-1] = "#fee2e2"
        cell_colors.append(row_c)

    table = ax_tab.table(cellText=cell_data,
                         cellColours=cell_colors,
                         colWidths=col_widths,
                         loc='center',
                         cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1.0, 1.7)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor('#cbd5e1')
        cell.set_linewidth(0.8)
        if row_idx == 0:
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
            cell.set_height(0.10)
        else:
            if col_idx == 0:
                cell.get_text().set_ha('left')
                cell.get_text().set_position((0.04, 0.5))
            if col_idx == len(col_headers) - 1:
                is_ok = table_rows[row_idx - 1]["aprovado"]
                cell.get_text().set_color('#15803d' if is_ok else '#b91c1c')
                cell.get_text().set_weight('bold')

    plt.subplots_adjust(top=0.87, bottom=0.04, left=0.06, right=0.94)
    _salvar_ou_mostrar("avaliacao_estatica_etapa5.png", salvar)

    return tudo_aprovado




# =============================================================================
# 6. AVALIAÇÃO ESTÁTICA DEDICADA: ETAPA 6 (ESW - BÓNUS OPCIONAL FUSÃO SENSORIAL)
# =============================================================================
def avaliar_etapa6(modulo, salvar=False):
    """
    Avaliação Estática Pura da Tarefa 6 (BÓNUS OPCIONAL - Fusão Sensorial GPS / Odometria):
    Gera uma missão contínua de 11.5 segundos contendo:
    1. Consenso nominal entre odometria estimada e antena GNSS/GPS (erro < 1.5m -> FUSION_OK)
    2. Glitch transitório de multipath por 0.2s (< 1.0s -> RETRY_RESET sem disparar EBS)
    3. Recuperação de sinal e reset do temporizador a 0.0s
    4. Divergência espacial sustentada por > 1.0s (erro > 1.5m -> SENSOR_FUSION_LOST com EBS ativo)
    5. Imobilização failsafe de emergência
    """
    print("\n" + "="*70)
    print(" AVALIAÇÃO ESTÁTICA DEDICADA - ETAPA 6: BÓNUS - FUSÃO SENSORIAL & FAIL-SAFE GPS")
    print("="*70)

    dt = 0.05
    time_vec = np.arange(0.0, 11.5, dt)
    N = len(time_vec)

    # 1. Trajetória da Odometria Estimada (Curva S contínua)
    x_est = 3.2 * time_vec
    y_est = 3.5 * np.sin(0.45 * time_vec)

    # 2. Dados Brutos da Antena GNSS/GPS
    x_gps = np.copy(x_est)
    y_gps = np.copy(y_est)

    gt_consenso = np.ones(N, dtype=bool)
    gt_glitch = np.zeros(N, dtype=bool)
    gt_divergencia = np.zeros(N, dtype=bool)
    gt_ebs = np.zeros(N, dtype=bool)

    np.random.seed(42)
    ruido_nominal = np.random.normal(0, 0.06, N)

    gt_tempo_div = 0.0
    for i, t in enumerate(time_vec):
        x_gps[i] += ruido_nominal[i]
        y_gps[i] += ruido_nominal[i]

        if 3.0 <= t < 3.2:
            # Glitch transitório (salto de multipath por 0.2s < 1.0s)
            y_gps[i] += 2.2
            gt_consenso[i] = False
            gt_glitch[i] = True
        elif 6.5 <= t < 9.5:
            # Divergência sustentada de GPS (> 1.5m por 3.0s > 1.0s)
            shift = min(3.0, 1.8 + 0.8 * (t - 6.5))
            y_gps[i] -= shift
            gt_consenso[i] = False
            gt_divergencia[i] = True

        dist_real = np.hypot(x_est[i] - x_gps[i], y_est[i] - y_gps[i])
        if dist_real > 1.5:
            gt_tempo_div += dt
        else:
            gt_tempo_div = 0.0
        gt_ebs[i] = (gt_tempo_div >= 1.0)

    # 3. Execução do algoritmo sob teste
    aluno_dist_err = np.zeros(N)
    aluno_tempo_div = np.zeros(N)
    aluno_estados = []
    aluno_ebs = np.zeros(N, dtype=bool)
    excecao_msg = None

    if hasattr(modulo, '_tempo_divergencia_gps'):
        modulo._tempo_divergencia_gps = 0.0

    if not hasattr(modulo, "supervisionar_fusao_gps"):
        excecao_msg = "Função 'supervisionar_fusao_gps' não implementada no módulo"

    for i in range(N):
        if excecao_msg is not None:
            break
        try:
            res = modulo.supervisionar_fusao_gps((x_est[i], y_est[i]), (x_gps[i], y_gps[i]), dt=dt)
            aluno_dist_err[i] = res.get('erro_distancia', 0.0)
            aluno_tempo_div[i] = res.get('tempo_divergencia', 0.0)
            st = res.get('estado_fusao', "DESCONHECIDO")
            aluno_estados.append(st)
            aluno_ebs[i] = res.get('ebs_trigger', False)
        except Exception as e:
            if excecao_msg is None:
                excecao_msg = str(e)
            aluno_dist_err[i] = 0.0
            aluno_tempo_div[i] = 0.0
            aluno_estados.append("ERRO")
            aluno_ebs[i] = False

    # 4. Avaliação dos Critérios
    if excecao_msg is not None:
        consenso_ok = False
        glitch_ok = False
        persistencia_ok = False
        ebs_ok = False
    else:
        # Consenso: estado FUSION_OK nos primeiros 2.5s
        consenso_ok = all(s == "FUSION_OK" for s in aluno_estados[:int(2.5/dt)])
        # Glitch: no pico de 0.2s transita para RETRY_RESET sem disparar EBS
        idx_glitch = np.where(gt_glitch)[0]
        glitch_ok = any(aluno_estados[i] == "RETRY_RESET" for i in idx_glitch) and not any(aluno_ebs[idx_glitch])
        # Persistência: transita para SENSOR_FUSION_LOST após 1.0s de divergência
        idx_div_crit = np.where(gt_ebs)[0]
        persistencia_ok = all(aluno_estados[i] == "SENSOR_FUSION_LOST" for i in idx_div_crit)
        # EBS: dispara no momento correto
        ebs_ok = all(aluno_ebs[idx_div_crit]) and not any(aluno_ebs[~gt_ebs])

    total_aprovados = sum([consenso_ok, glitch_ok, persistencia_ok, ebs_ok])
    tudo_aprovado = (total_aprovados == 4) and (excecao_msg is None)

    print(f"-> Consenso Nominal (Erro < 1.5m): {'[APROVADO]' if consenso_ok else '[FALHOU]'}")
    print(f"-> Rejeição de Glitch Transitório (< 1.0s): {'[APROVADO]' if glitch_ok else '[FALHOU]'}")
    print(f"-> Gatilho de Persistência Temporal (>= 1.0s): {'[APROVADO]' if persistencia_ok else '[FALHOU]'}")
    print(f"-> Disparo Failsafe de EBS: {'[APROVADO]' if ebs_ok else '[FALHOU]'}\n")

    # 5. Construção da Figura Visual
    fig = plt.figure(figsize=(15.5, 10.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.0, 1.0], hspace=0.38)
    fig.suptitle("ETAPA 6 [ESW - BÓNUS OPCIONAL]: Supervisão de Fusão Sensorial & Fail-Safe GPS", 
                 fontsize=13, fontweight='bold', y=0.978)

    # Top Banner
    resumo_color = '#15803d' if tudo_aprovado else '#b91c1c'
    resumo_bg = '#f0fdf4' if tudo_aprovado else '#fef2f2'
    resumo_border = '#86efac' if tudo_aprovado else '#fca5a5'
    resumo_texto = (
        f"Resultado da Avaliação Bónus: {total_aprovados}/4 Critérios Aprovados   |   "
        f"Consenso Nominal: {'OK' if consenso_ok else 'FAIL'}   •   "
        f"Glitch Transitório (<1.0s): {'OK' if glitch_ok else 'FAIL'}   •   "
        f"Persistência Temporal (1.0s): {'OK' if persistencia_ok else 'FAIL'}   •   "
        f"Disparo EBS: {'OK' if ebs_ok else 'FAIL'}"
    )
    fig.text(0.5, 0.942, resumo_texto, ha='center', va='center', fontsize=9.2, fontweight='bold', 
             color=resumo_color, bbox=dict(boxstyle='round,pad=0.38', facecolor=resumo_bg, edgecolor=resumo_border, linewidth=1.2))

    # --- SUBPLOT 1: Trajetória no Plano 2D (Odometria vs GPS) ---
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title("1. Trajetória 2D: Odometria Estimada vs Antena GNSS/GPS", 
                  fontsize=9.5, fontweight='bold', loc='left', pad=10)
    l_odo, = ax1.plot(x_est, y_est, color='black', linewidth=2.2, label='Odometria / SLAM (Estimativa)')
    l_gps, = ax1.plot(x_gps, y_gps, color='#d946ef', linestyle='--', linewidth=1.6, label='Coordenadas GPS Brutas')
    
    ax1.scatter(x_gps[gt_glitch], y_gps[gt_glitch], color='#f59e0b', s=35, zorder=5, label='Glitch Transitório (0.2s)')
    ax1.scatter(x_gps[gt_ebs], y_gps[gt_ebs], color='#dc2626', s=35, zorder=5, label='Divergência Crítica (>1.0s)')

    ax1.set_xlabel("Coordenada X (metros)", fontsize=8.5, fontweight='bold')
    ax1.set_ylabel("Coordenada Y (metros)", fontsize=8.5, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='lower right', bbox_to_anchor=(1.0, 1.01), ncol=4, fontsize=8, frameon=False)

    # --- SUBPLOT 2: Série Temporal de Erro Euclidiano de Posição ---
    ax2 = fig.add_subplot(gs[1], sharex=None)
    ax2.set_title("2. Erro de Distância Euclidiana entre Sensores vs Limiar de Desagregação Espacial", 
                  fontsize=9.5, fontweight='bold', loc='left', pad=10)
    l_err, = ax2.plot(time_vec, aluno_dist_err, color='#2563eb', linewidth=2.0, label='Erro de Distância ||Pos_Est - Pos_GPS||')
    l_lim_esp = ax2.axhline(1.5, color='#dc2626', linestyle='--', linewidth=1.4, alpha=0.85, label='Limiar Espacial (1.5 metros)')

    ax2.text(3.1, 2.35, "Pico Transitório\n(0.2s < 1.0s: Rejeitar)", color='#d97706', fontsize=7.5, ha='center', fontweight='bold')
    ax2.text(8.0, 3.15, "Divergência Sustentada\n(>= 1.0s: Falha Crítica)", color='#dc2626', fontsize=7.5, ha='center', fontweight='bold')

    ax2.set_ylabel("Erro (metros)", fontsize=8.5, fontweight='bold')
    ax2.set_ylim(-0.2, 3.8)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='lower right', bbox_to_anchor=(1.0, 1.01), ncol=2, fontsize=8, frameon=False)

    # --- SUBPLOT 3: Canais Digitais de Fusão e EBS (Logic Analyzer) ---
    ax3 = fig.add_subplot(gs[2], sharex=ax2)
    ax3.set_title("3. Canais Digitais de Supervisão da Fusão Sensorial e Disparo Failsafe de EBS", 
                  fontsize=9.5, fontweight='bold', loc='left', pad=10)
    ax3.text(1.0, 1.03, "Faixa 1: EBS Ativado  |  Faixa 2: Estado da Fusão Sensorial", 
             transform=ax3.transAxes, ha='right', fontsize=8, color='#475569', fontstyle='italic')

    # Faixa 1 (Inferior): EBS Trigger (0 ou 1)
    ax3.axhspan(-0.15, 0.95, color='#fef2f2', alpha=0.35)
    ax3.axvspan(7.5, 9.5, ymin=0.0/2.3, ymax=1.0/2.3, color='#fca5a5', alpha=0.55)
    ebs_line_color = '#dc2626' if ebs_ok else '#991b1b'

    # Faixa 2 (Superior): Estado da Fusão (0=FUSION_OK, 1=RETRY_RESET, 2=SENSOR_FUSION_LOST)
    ax3.axhspan(1.15, 2.35, color='#eff6ff', alpha=0.35)
    
    if excecao_msg is not None:
        ax3.text(5.75, 1.15, f"[TAREFA BÓNUS OPCIONAL NÃO IMPLEMENTADA: {excecao_msg}]", 
                 color='darkred', ha='center', va='center', fontsize=10.5, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#fee2e2', edgecolor='red', alpha=0.9))
    else:
        map_estados = {"FUSION_OK": 1.35, "RETRY_RESET": 1.75, "SENSOR_FUSION_LOST": 2.15}
        y_estados = np.array([map_estados.get(s, 1.35) for s in aluno_estados])
        ax3.step(time_vec, aluno_ebs.astype(int) * 0.70 + 0.10, where='post', color=ebs_line_color, linewidth=2.2)
        ax3.step(time_vec, y_estados, where='post', color='#2563eb', linewidth=2.2)

    ax3.set_yticks([0.10, 0.80, 1.35, 1.75, 2.15])
    ax3.set_yticklabels(['OFF', 'ATIVO', 'OK', 'RESET', 'LOST'], fontsize=7.5)

    ax3.text(-0.025, 0.45, "CANAL 1:\nEBS_TRIGGER", transform=ax3.get_yaxis_transform(), 
             ha='right', va='center', fontsize=8, fontweight='bold', color='#dc2626')
    ax3.text(-0.025, 1.75, "CANAL 2:\nESTADO FUSÃO", transform=ax3.get_yaxis_transform(), 
             ha='right', va='center', fontsize=8, fontweight='bold', color='#2563eb')

    ax3.set_xlabel("Tempo de Simulação (segundos)", fontsize=9.0, fontweight='bold')
    ax3.set_xlim(0, 11.5)
    ax3.set_ylim(-0.15, 2.35)
    ax3.grid(True, linestyle=':', alpha=0.5)

    plt.subplots_adjust(top=0.90, bottom=0.06, left=0.13, right=0.94)
    _salvar_ou_mostrar("avaliacao_estatica_etapa6.png", salvar)

    return tudo_aprovado


# =============================================================================
# ENTRADA PRINCIPAL
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Avaliação Estática Formula Student FSUminho")
    parser.add_argument("--solucao", action="store_true", help="Usa a solução de referência (gabarito 100%%)")
    parser.add_argument("--etapa", type=int, choices=[1, 2, 3, 4, 5, 6], default=1, help="Avaliar apenas uma etapa específica (1 a 6)")
    parser.add_argument("--salvar", action="store_true", help="Guarda a imagem gerada em ficheiro PNG sem abrir janela GUI")
    args = parser.parse_args()

    # Importar módulo a avaliar
    if args.solucao:
        import solucao as modulo
        print("[MODO]: A testar SOLUÇÃO DE REFERÊNCIA (solucao.py)")
    else:
        try:
            import desafio_aluno as modulo
            print("[MODO]: A testar CÓDIGO DO ALUNO (desafio_aluno.py)")
        except Exception as e:
            print(f"[ERRO]: Não foi possível importar desafio_aluno.py: {e}")
            sys.exit(1)

    track_gen = TrackGenerator(track_width=3.2, cone_spacing=2.8, seed=42)

    if args.etapa == 1:
        avaliar_etapa1(modulo, track_gen, salvar=args.salvar)
    elif args.etapa == 2:
        avaliar_etapa2(modulo, track_gen, salvar=args.salvar)
    elif args.etapa == 3:
        avaliar_etapa3(modulo, track_gen, salvar=args.salvar)
    elif args.etapa == 4:
        avaliar_etapa4(modulo, salvar=args.salvar)
    elif args.etapa == 5:
        avaliar_etapa5(modulo, salvar=args.salvar)
    elif args.etapa == 6:
        avaliar_etapa6(modulo, salvar=args.salvar)


if __name__ == "__main__":
    main()
