import sys
import cv2 
import numpy as np
import yaml
import os
from termcolor import cprint

class StereoVision:
    # Configuración por defecto (Fallback si no hay YAML)
    DEFAULT_CONFIG = {
        "stereo": {
            "min_disparity": 0,
            "num_disparities": 128, # Debe ser divisible por 16
            "block_size": 5,
            "p1_factor": 8,   # P1 = 8 * 3 * block_size^2
            "p2_factor": 32,  # P2 = 32 * 3 * block_size^2
            "disp12_max_diff": 1,
            "uniqueness_ratio": 10,
            "speckle_window_size": 100,
            "speckle_range": 32,
            "mode": "SGBM_3WAY" # Options: SGBM, HH, SGBM_3WAY
        },
        "wls": {
            "lambda": 8000.0,
            "sigma": 1 # Bajo para preservar bordes!!
        }
    }

    def __init__(self, calibration_data=None, config_path=None):
        """
        Inicializa el motor estéreo.
        Args:
            calibration_data: Diccionario con matrices (Q, P, etc.)
            config_path: Ruta a un archivo .yaml con la configuración.
        """
        self.calibration = calibration_data
        
        # 1. Cargar Configuración
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_cfg = yaml.safe_load(f)
                # Actualizar recursivamente (simple)
                if loaded_cfg:
                    for section in ['stereo', 'wls']:
                        if section in loaded_cfg:
                            self.config[section].update(loaded_cfg[section])
            print(f"✅ Configuración estéreo cargada de: {config_path}")
        else:
            print("⚠️ Usando configuración estéreo por defecto.")
        
        cprint("La config es:", "green")
        cprint("-" * 40,"green")
        print(yaml.dump(self.config, default_flow_style=False, sort_keys=False),"green")
        print("-" * 40,"green")

        # Atajos para legibilidad
        s_cfg = self.config['stereo']
        w_cfg = self.config['wls']
        
        self.num_disp = s_cfg['num_disparities']
        self.block_size = s_cfg['block_size']
        
        # Mapeo de modos
        mode_map = {
            "SGBM": cv2.STEREO_SGBM_MODE_SGBM,
            "HH": cv2.STEREO_SGBM_MODE_HH,
            "SGBM_3WAY": cv2.STEREO_SGBM_MODE_SGBM_3WAY
        }
        mode = mode_map.get(s_cfg.get('mode', 'SGBM_3WAY'), cv2.STEREO_SGBM_MODE_SGBM_3WAY)

        # 2. Crear Left Matcher (SGBM)
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

        # 3. Crear Right Matcher y WLS Filter (Se crean siempre, se usan bajo demanda)
        # Esto requiere opencv-contrib-python
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
        self.wls_filter.setLambda(w_cfg['lambda'])
        self.wls_filter.setSigmaColor(w_cfg['sigma'])
        self.has_wls = True

    def compute_disparity(self,frame_id ,img_l, img_r, strips, use_wls=False, debug=False, debug_path=None):
        """
        Calcula el mapa de disparidad.
        Args:
            strips: Lista de tuplas (y1, y2) para procesar solo franjas.
            use_wls (bool): INTERRUPTOR para activar/desactivar el filtrado WLS.
            debug (bool): Si True, muestra en pantalla.
            debug_path (str): Si se da una ruta, guarda la imagen de debug ahí.
        """
        h, w = img_l.shape[:2]
        full_map = np.zeros((h, w), dtype=np.float32)
        
        if not strips:
            return full_map

        # Chequeo de seguridad
        do_wls = use_wls and self.has_wls

        for y1, y2 in strips:
            # 1. Recorte (Crop)
            strip_l = img_l[y1:y2, :]
            strip_r = img_r[y1:y2, :]
            
            # 2. Cálculo SGBM
            if do_wls:
                # Camino lento pero preciso (WLS)
                disp_l = self.left_matcher.compute(strip_l, strip_r)
                disp_r = self.right_matcher.compute(strip_r, strip_l)
                
                disp_filtered = self.wls_filter.filter(
                    disparity_map_left=disp_l,
                    left_view=strip_l, # WLS usa la imagen original para guiar el suavizado
                    disparity_map_right=disp_r
                )
                
                # Normalización (x16 -> float)
                disp_float = disp_filtered.astype(np.float32) / 16.0
                
            else:
                # Camino rápido (Solo SGBM)
                raw_disp = self.left_matcher.compute(strip_l, strip_r)
                disp_float = raw_disp.astype(np.float32) / 16.0
            
            # 3. Limpieza (Quitar valores negativos o inválidos)
            disp_float[disp_float < 0] = 0
            
            # 4. Pegar en el mapa final
            full_map[y1:y2, :] = disp_float
            
            
            
            if debug or (debug_path is not None):
                
                disp_img_path = os.path.join(debug_path,frame_id+"_disparity.png")
                self.visualize_disparity(
                    full_map, 
                    strips=strips, 
                    show=debug,          # Mostrar solo si debug=True
                    save_path=disp_img_path # Guardar solo si hay ruta
                )
            
        return full_map
    
    
    def visualize_disparity(self, disparity_map, strips=None, show=False, save_path=None, window_name="Stereo Debug"):
        """
        Genera una visualización coloreada del mapa de disparidad.
        
        Args:
            disparity_map: Imagen float32.
            strips: Lista de franjas para dibujar líneas (opcional).
            show (bool): Si True, abre una ventana de OpenCV.
            save_path (str): Si no es None, guarda la imagen en esa ruta.
            window_name (str): Nombre de la ventana.
            
        Returns:
            vis_color: La imagen generada (BGR).
        """
        # 1. Normalizar y Colorear
        norm_disp = (disparity_map / self.num_disp) * 255.0
        vis_uint8 = np.clip(norm_disp, 0, 255).astype(np.uint8)
        vis_color = cv2.applyColorMap(vis_uint8, cv2.COLORMAP_JET)
        
        # Poner en negro los datos inválidos (0)
        vis_color[disparity_map == 0] = 0
        
        # 2. Dibujar Franjas
        if strips:
            h, w = disparity_map.shape[:2]
            for y1, y2 in strips:
                cv2.line(vis_color, (0, y1), (w, y1), (0, 255, 0), 1)
                cv2.line(vis_color, (0, y2), (w, y2), (0, 255, 0), 1)

        # 3. OPCIÓN: GUARDAR
        if save_path is not None:
            # Crear carpeta si no existe (evita errores)
            import os
            directory = os.path.dirname(save_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                
            cv2.imwrite(save_path, vis_color)
            # print(f"💾 Disparity map saved: {save_path}")

        # 4. OPCIÓN: MOSTRAR
        if show:
            cv2.imshow(window_name, vis_color)
            cv2.waitKey(1) # 1ms para refrescar ventana
            
        return vis_color
    
    def get_decimated_Q():
        return np.float32([
            [1, 0, 0, -CX],
            [0, 1, 0, -CY],
            [0, 0, 0, FOCAL_DECIMATED],
            [0, 0, 1.0/BASELINE, 0]  # Positivo para Z hacia adelante
        ])

    def reproject_to_3d(self, disparity_map, roi_mask=None):
        """Genera nubes de puntos XYZ usando la matriz Q."""
        if self.calibration is None or 'Q' not in self.calibration:
             # Si no hay Q, no podemos reproyectar metricamente
             return None
        #TODO ! LA Q HA DE SER DECIMATED SI USO LES DECIMATED!!
        points_3d = cv2.reprojectImageTo3D(disparity_map, self.calibration['Q'])
        
        if roi_mask is not None:
            return points_3d[roi_mask > 0]
            
        return points_3d