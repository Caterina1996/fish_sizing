import sys
import cv2 
import numpy as np
import yaml
import os
import open3d as o3d 

class StereoVision:
    # Configuración por defecto
    DEFAULT_CONFIG = {
        "stereo": {
            "min_disparity": 0,
            "num_disparities": 128,
            "block_size": 5,
            "p1_factor": 8,
            "p2_factor": 32,
            "disp12_max_diff": 1,
            "uniqueness_ratio": 10,
            "speckle_window_size": 100,
            "speckle_range": 32,
            "mode": "SGBM_3WAY"
        },
        "wls": {
            "lambda": 8000.0,
            "sigma": 1
        }
    }

    def __init__(self, calibration_data, config_path=None, scale=0.5):
        self.calibration = calibration_data
        
        # 1. Extracción Dinámica de Constantes
        try:
            def get_P(info):
                if hasattr(info, 'P'): return np.array(info.P).reshape(3,4)
                return np.array(info['projection_matrix']['data']).reshape(3,4)

            P_left = get_P(self.calibration['left'])
            P_right = get_P(self.calibration['right'])

            raw_fx = P_left[0, 0]
            raw_cx = P_left[0, 2]
            raw_cy = P_left[1, 2]

            self.FOCAL = raw_fx * scale
            self.CX = raw_cx * scale
            self.CY = raw_cy * scale

            tx_right = P_right[0, 3]
            self.BASELINE = abs(tx_right / raw_fx)

            print(f"📐 CALIBRACIÓN AUTOMÁTICA (Escala {scale}x):")
            print(f"   -> Focal: {self.FOCAL:.2f} px")
            print(f"   -> Baseline: {self.BASELINE:.4f} m")

        except Exception as e:
            print(f"❌ Error leyendo calibración: {e}. Usando default.")
            self.FOCAL = 730.35 
            self.BASELINE = 0.1203
            self.CX = 520.70
            self.CY = 380.34

        # 3. Cargar Configuración YAML
        self.config = self.DEFAULT_CONFIG.copy()
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_cfg = yaml.safe_load(f)
                if loaded_cfg:
                    for section in ['stereo', 'wls']:
                        if section in loaded_cfg:
                            self.config[section].update(loaded_cfg[section])
                            
        # Configurar Filtro Profundidad -> més enllà d'aquesta distància no em crec l'stereo
        self.max_depth_meters = self.config['stereo']['max_depth_meters']  
        if self.max_depth_meters > 0:
            self.min_valid_disparity = (self.FOCAL * self.BASELINE) / self.max_depth_meters
        else:
            self.min_valid_disparity = 0

        # 4. Crear Matchers
        s_cfg = self.config['stereo']
        w_cfg = self.config['wls']
        
        self.num_disp = s_cfg['num_disparities']
        self.block_size = s_cfg['block_size']
        
        mode_map = {"SGBM": 0, "HH": 1, "SGBM_3WAY": 2}
        mode = mode_map.get(s_cfg.get('mode', 'SGBM_3WAY'), 2)

        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=s_cfg['min_disparity'],
            numDisparities=self.num_disp,
            blockSize=self.block_size,
            P1=s_cfg['p1_factor'] * 3 * self.block_size**2,
            P2=s_cfg['p2_factor'] * 3 * self.block_size**2,
            disp12MaxDiff=s_cfg['disp12_max_diff'],
            uniquenessRatio=s_cfg['uniqueness_ratio'],
            speckleWindowSize=s_cfg['speckle_window_size'],
            speckleRange=s_cfg['speckle_range'],
            mode=mode
        )

        try:
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
            self.wls_filter.setLambda(w_cfg['lambda'])
            self.wls_filter.setSigmaColor(w_cfg['sigma'])
            self.has_wls = True
        except AttributeError:
            self.has_wls = False

        # 5. IMPORTANTE: Generar la Matriz Q
        self.calibration['Q'] = self.get_decimated_Q()

    def compute_disparity(self, frame_id, img_l, img_r, strips, use_wls=False, debug=False, debug_path=None):
        h, w = img_l.shape[:2]
        full_map = np.zeros((h, w), dtype=np.float32)
        
        if not strips: return full_map

        do_wls = use_wls and self.has_wls

        for y1, y2 in strips:
            strip_l = img_l[y1:y2, :]
            strip_r = img_r[y1:y2, :]
            
            if do_wls:
                disp_l = self.left_matcher.compute(strip_l, strip_r)
                disp_r = self.right_matcher.compute(strip_r, strip_l)
                disp_filtered = self.wls_filter.filter(disp_l, strip_l, disparity_map_right=disp_r)
                disp_float = disp_filtered.astype(np.float32) / 16.0
            else:
                raw_disp = self.left_matcher.compute(strip_l, strip_r)
                disp_float = raw_disp.astype(np.float32) / 16.0
            
            disp_float[disp_float < 0] = 0
            
            # Filtro físico
            valid_disp_mask = (disp_float > self.min_valid_disparity)
            disp_float[~valid_disp_mask] = 0
            
            full_map[y1:y2, :] = disp_float
            
            if debug or (debug_path is not None):
                filename = f"{frame_id}_disparity.png"
                save_p = os.path.join(debug_path, filename) if debug_path else None
                self.visualize_disparity(full_map, strips, show=debug, save_path=save_p)
            
        return full_map
    
    def visualize_disparity(self, disparity_map, strips=None, show=False, save_path=None):
        norm_disp = (disparity_map / self.num_disp) * 255.0
        vis_uint8 = np.clip(norm_disp, 0, 255).astype(np.uint8)
        vis_color = cv2.applyColorMap(vis_uint8, cv2.COLORMAP_JET)
        vis_color[disparity_map == 0] = 0
        
        if strips:
            h, w = disparity_map.shape[:2]
            for y1, y2 in strips:
                cv2.line(vis_color, (0, y1), (w, y1), (0, 255, 0), 1)
                cv2.line(vis_color, (0, y2), (w, y2), (0, 255, 0), 1)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, vis_color)

        if show:
            cv2.imshow("Debug", vis_color)
            cv2.waitKey(1)
            
        return vis_color
    
    def get_decimated_Q(self):
        return np.float32([
            [1, 0, 0, -self.CX],
            [0, 1, 0, -self.CY],
            [0, 0, 0, self.FOCAL],
            [0, 0, 1.0/self.BASELINE, 0]
        ])

    def reproject_to_3d(self, disparity_map):
        """
        Calcula la nube de puntos completa (H, W, 3).
        """
        # Genera una imagen (H, W, 3) con coordenadas X, Y, Z
        self.points_3d = cv2.reprojectImageTo3D(disparity_map, self.calibration['Q'])        
        return self.points_3d

    def save_point_cloud(self,points_3d, colors, mask, save_path, z_min=0.1, z_max = None):
        """
        Guarda un PLY filtrando por máscara y profundidad.
        """
        # Pasar z_max si quiero usar una distancia maxima distinta
        if z_max == None:
            z_max = self.max_depth_meters
        
        # Si no pasan mascara devolvemos la pc de toda la franja    
        if mask is None:
            mask = np.ones(points_3d.shape[:2], dtype=bool)

        valid_points = points_3d[mask > 0]
        valid_colors = colors[mask > 0]
        
        if len(valid_points) == 0:
            print(f"[WARN] Nube vacía para {os.path.basename(save_path)}")
            return

        # 2. Filtrar por profundidad (Z) y limpiar infinitos
        zs = valid_points[:, 2]
        # isfinite quita los 'inf' que genera OpenCV donde disp=0
        z_mask = (zs > z_min) & (zs < z_max) & np.isfinite(zs)
        
        final_points = valid_points[z_mask]
        final_colors = valid_colors[z_mask]
        
        if len(final_points) > 0:
            print(f"   💾 PLY: {os.path.basename(save_path)} ({len(final_points)} pts)")
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(final_points)
            # OpenCV es BGR -> Open3D es RGB (Hacemos flip ::-1)
            pcd.colors = o3d.utility.Vector3dVector(final_colors[:, ::-1] / 255.0)
            
            # Crear carpeta si no existe
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            o3d.io.write_point_cloud(save_path, pcd)
            