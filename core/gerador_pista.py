"""
=============================================================================
FSUminho AS / ESW Challenge - Gerador de Pista e Simulação de Sensores
=============================================================================
Gera uma pista oficial Formula Student Driverless (FSD):
- Limite esquerdo: Cones Azuis
- Limite direito: Cones Amarelos
- Linha de Meta/Partida: Cones Grandes Laranjas
- Cones com ruído realista e falsos positivos para filtrar na perceção.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
from typing import List, Tuple, Dict
from contrato_dados import Cone

class TrackGenerator:
    def __init__(self, track_width=3.2, cone_spacing=2.8, seed=42):
        self.track_width = track_width
        self.cone_spacing = cone_spacing
        self.seed = seed
        np.random.seed(seed)
        
        self.midline_x, self.midline_y = None, None
        self.blue_cones: List[Cone] = []
        self.yellow_cones: List[Cone] = []
        self.orange_cones: List[Cone] = []
        self.all_cones_ground_truth: List[Cone] = []
        
        self._build_circuit()

    def _build_circuit(self):
        # Pontos de controlo do circuito Formula Student com curvas técnicas de menor raio
        # Inclui chicanes rápidas, hairpin fechado e cotovelo de 90 graus
        waypoints = np.array([
            [0, 15], [10, 35], [5, 55], [20, 70], [45, 65],
            [55, 45], [40, 35], [55, 20], [80, 25], [100, 45],
            [110, 65], [130, 70], [145, 50], [135, 25], [150, 10],
            [130, -15], [115, -10], [95, -30], [70, -35], [50, -15],
            [35, -30], [15, -25], [-5, -5]
        ])

        tck, u = splprep([waypoints[:, 0], waypoints[:, 1]], s=2, per=1)
        points_count = 1500
        u_fine = np.linspace(0, 1, points_count)
        x_spline, y_spline = splev(u_fine, tck)
        
        # Assegurar ciclo fechado
        x_spline[-1] = x_spline[0]
        y_spline[-1] = y_spline[0]
        
        self.midline_x = x_spline
        self.midline_y = y_spline
        
        # Vetores tangentes e normais
        dx = np.gradient(x_spline)
        dy = np.gradient(y_spline)
        ds = np.sqrt(dx**2 + dy**2)
        
        # Vetor normal unitário apontando para fora da curva
        nx = -dy / ds
        ny = dx / ds
        
        # Colocar cones ao longo da pista com espaçamento
        s_accum = np.cumsum(ds)
        total_length = s_accum[-1]
        num_cones = int(total_length // self.cone_spacing)
        cone_indices = np.linspace(0, len(x_spline) - 2, num_cones, dtype=int)
        
        half_w = self.track_width / 2.0
        
        for idx in cone_indices:
            # Posição central
            xc, yc = x_spline[idx], y_spline[idx]
            norm_x, norm_y = nx[idx], ny[idx]
            
            # Pequena perturbação realista na colocação de cones (+- 5 cm)
            jitter_l = np.random.normal(0, 0.05, 2)
            jitter_r = np.random.normal(0, 0.05, 2)
            
            # Cones Azuis (Esquerda)
            bx = xc + norm_x * half_w + jitter_l[0]
            by = yc + norm_y * half_w + jitter_l[1]
            self.blue_cones.append(Cone(bx, by, 'blue', confidence=0.98))
            
            # Cones Amarelos (Direita)
            yx = xc - norm_x * half_w + jitter_r[0]
            yy = yc - norm_y * half_w + jitter_r[1]
            self.yellow_cones.append(Cone(yx, yy, 'yellow', confidence=0.98))

        # Cones Laranja (Linha de Partida / Chegada)
        start_idx = 0
        xc, yc = x_spline[start_idx], y_spline[start_idx]
        norm_x, norm_y = nx[start_idx], ny[start_idx]
        
        self.orange_cones.append(Cone(xc + norm_x * (half_w + 0.3), yc + norm_y * (half_w + 0.3), 'orange', 1.0))
        self.orange_cones.append(Cone(xc - norm_x * (half_w + 0.3), yc - norm_y * (half_w + 0.3), 'orange', 1.0))
        
        # Artefactos Físicos Permanentes ao redor da pista (barreiras de pneus, fardos de palha, postes)
        # Ficam fixos no mapa e fora dos limites da pista (offset lateral de 3.5m a 6.0m)
        self.physical_artifacts = []
        artifact_indices = np.linspace(15, len(x_spline) - 15, 30, dtype=int)
        for i, idx in enumerate(artifact_indices):
            xc, yc = x_spline[idx], y_spline[idx]
            norm_x, norm_y = nx[idx], ny[idx]
            side = 1.0 if (i % 2 == 0) else -1.0
            offset = np.random.uniform(3.2, 5.5)
            art_x = xc + norm_x * (half_w + offset) + np.random.normal(0, 0.2)
            art_y = yc + norm_y * (half_w + offset) + np.random.normal(0, 0.2)
            self.physical_artifacts.append((float(art_x), float(art_y)))

        self.all_cones_ground_truth = self.blue_cones + self.yellow_cones + self.orange_cones

    def get_ground_truth_track(self):
        """Retorna listas de coordenadas dos cones reais, artefactos físicos e da midline"""
        return {
            'blue': [(c.x, c.y) for c in self.blue_cones],
            'yellow': [(c.x, c.y) for c in self.yellow_cones],
            'orange': [(c.x, c.y) for c in self.orange_cones],
            'artifacts': list(self.physical_artifacts),
            'midline': list(zip(self.midline_x, self.midline_y))
        }

    def simulate_sensor_detections(self, car_x: float, car_y: float, car_yaw: float, 
                                   max_range=14.0, fov_deg=120.0, add_noise=True,
                                   extra_noise=False, drop_side=None) -> List[Dict]:
        """
        Simula a perceção (LiDAR / Câmara):
        Retorna cones visíveis com ruído gaussiano e falsos positivos espúrios.
        - extra_noise=True: injeta ruído severo para testar a Etapa 1
        - drop_side='blue'/'yellow': omite um lado para testar edge cases na Etapa 2
        """
        detections = []
        effective_range = max_range * 1.3 if extra_noise else max_range
        fov_rad = np.radians(fov_deg)
        half_fov = fov_rad / 2.0
        
        for cone in self.all_cones_ground_truth:
            # Omitir um lado se solicitado (edge case de curva cega na Etapa 2)
            if drop_side == 'blue' and cone.color == 'blue':
                continue
            if drop_side == 'yellow' and cone.color == 'yellow':
                continue

            dx = cone.x - car_x
            dy = cone.y - car_y
            dist = np.hypot(dx, dy)
            
            if dist > effective_range:
                continue
                
            angle_global = np.arctan2(dy, dx)
            angle_rel = (angle_global - car_yaw + np.pi) % (2 * np.pi) - np.pi
            
            if abs(angle_rel) <= half_fov:
                if add_noise:
                    meas_x = cone.x + np.random.normal(0, 0.12 if extra_noise else 0.08)
                    meas_y = cone.y + np.random.normal(0, 0.12 if extra_noise else 0.08)
                    
                    dist_factor = max(0.4, 1.0 - (dist / effective_range) * 0.5)
                    conf = float(np.clip(cone.confidence * dist_factor + np.random.normal(0, 0.08 if extra_noise else 0.04), 0.1, 1.0))
                    
                    color = cone.color
                    # Chance de classificar como unknown
                    unknown_chance = 0.15 if extra_noise else 0.03
                    if np.random.rand() < unknown_chance:
                        color = 'unknown'
                else:
                    meas_x, meas_y = cone.x, cone.y
                    conf = cone.confidence
                    color = cone.color
                
                detections.append({
                    'x': float(meas_x),
                    'y': float(meas_y),
                    'color': color,
                    'confidence': float(conf)
                })
                
        # Deteção dos Artefactos Físicos Permanentes se estiverem no FOV do LiDAR/Câmara
        for art_x, art_y in self.physical_artifacts:
            dx = art_x - car_x
            dy = art_y - car_y
            dist = np.hypot(dx, dy)
            if dist > effective_range:
                continue
            angle_global = np.arctan2(dy, dx)
            angle_rel = (angle_global - car_yaw + np.pi) % (2 * np.pi) - np.pi
            if abs(angle_rel) <= half_fov:
                meas_x = art_x + np.random.normal(0, 0.10)
                meas_y = art_y + np.random.normal(0, 0.10)
                # Objeto físico sólido -> Alta confiança geométrica de reflexão LiDAR (0.75 a 0.98)
                conf = float(np.random.uniform(0.75, 0.96))
                detections.append({
                    'x': float(meas_x),
                    'y': float(meas_y),
                    'color': 'unknown', # O classificador ótico não reconhece como cone azul/amarelo!
                    'confidence': conf,
                    'type': 'physical_artifact'
                })

        # Adicionar ruído puramente de sensor (reflexões espúrias / flare / poeira transitória)
        # Este ruído só surge se add_noise=True e tem SEMPRE baixa confiança (< 0.60)
        if add_noise:
            num_spurious = np.random.randint(4, 8) if extra_noise else (1 if np.random.rand() < 0.25 else 0)
            for _ in range(num_spurious):
                rand_dist = np.random.uniform(2.5, effective_range)
                rand_ang = np.random.uniform(-half_fov, half_fov)
                ang_glob = car_yaw + rand_ang
                spurious_x = car_x + rand_dist * np.cos(ang_glob)
                spurious_y = car_y + rand_dist * np.sin(ang_glob)
                
                detections.append({
                    'x': float(spurious_x),
                    'y': float(spurious_y),
                    'color': np.random.choice(['blue', 'yellow', 'unknown']),
                    'confidence': float(np.random.uniform(0.12, 0.52)), # Baixa confiança! (< 0.60)
                    'type': 'sensor_noise' # Ruído transitório
                })
            
        return detections

if __name__ == "__main__":
    track = TrackGenerator()
    gt = track.get_ground_truth_track()
    print(f"Pista gerada com sucesso:")
    print(f"- Cones azuis: {len(gt['blue'])}")
    print(f"- Cones amarelos: {len(gt['yellow'])}")
    print(f"- Cones laranjas: {len(gt['orange'])}")
    print(f"- Pontos de midline: {len(gt['midline'])}")
    
    # Teste de deteção
    sample_det = track.simulate_sensor_detections(0, 15, 0)
    print(f"- Cones detetados no início: {len(sample_det)}")
