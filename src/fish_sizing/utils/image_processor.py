import cv2
import numpy as np
from termcolor import cprint

class ImageProcessor:
    def __init__(self, left_image=None, info_l=None, info_r=None , right_image=None, stereo=True):
        """
        Implement different options for image processing and enhacing
        """
        # Sava original images
        self.original_left = left_image
        self.original_right = right_image
        
        self.processed_left= None
        self.processed_right= None
        
        self.info_l = info_l
        self.info_r = info_r 
        
        self.rect_maps = None
                
        if info_l!= None and info_r!= None:
            self.generate_rectification_maps()
        
        self.is_stereo = stereo

        # 2. Initialize the processed ones
        
        if left_image!=None:
            self.processed_left = self.original_left.copy()
            
        if right_image!=None:
            self.processed_right = self.original_right.copy()

    
    def set_camera_info(self,info_l, info_r):
        self.info_l = info_l
        self.info_r = info_r 
    
    def set_image_pair(self,left_image,right_image):
        self.original_left = left_image
        self.original_right = right_image
        
        self.processed_left = self.original_left.copy()
        self.processed_right = self.original_right.copy()
        
            
    def reset(self):
        """Go back to original images"""
        self.processed_left = self.original_left.copy()
        if self.is_stereo:
            self.processed_right = self.original_right.copy()
        return self

    def get_processed(self):
        """Devuelve las imágenes en su estado actual de procesado."""
        if self.is_stereo:
            return self.processed_left, self.processed_right
        return self.processed_left
    
    def get_originals(self):
        """Devuelve las imágenes originales sin tocar."""
        if self.is_stereo:
            return self.original_left, self.original_right
        return self.original_left
    
    def generate_rectification_maps(self):
        """
        Genera los mapas de rectificación a partir de los mensajes ROS CameraInfo.
        Se llama UNA sola vez al principio del programa.
        
        Returns:
            Tupla: ((m_l1, m_l2), (m_r1, m_r2))
        """
        # Conversión de matrices (Tu código original)
        K_l, D_l = np.array(self.info_l.K).reshape(3,3), np.array(self.info_l.D)
        R_l, P_l = np.array(self.info_l.R).reshape(3,3), np.array(self.info_l.P).reshape(3,4)
        
        K_r, D_r = np.array(self.info_r.K).reshape(3,3), np.array(self.info_r.D)
        R_r, P_r = np.array(self.info_r.R).reshape(3,3), np.array(self.info_r.P).reshape(3,4)
        
        size = (self.info_l.width, self.info_l.height)

        # Generar mapas (Pesado, hacer solo una vez)
        self.m_l1, self.m_l2 = cv2.initUndistortRectifyMap(K_l, D_l, R_l, P_l, size, cv2.CV_16SC2)
        self.m_r1, self.m_r2 = cv2.initUndistortRectifyMap(K_r, D_r, R_r, P_r, size, cv2.CV_16SC2)
        
        self.rect_maps = ((self.m_l1, self.m_l2), (self.m_r1, self.m_r2))

        return self.rect_maps
    
    def rectify(self):
        """
        Aplica la rectificación geométrica usando los mapas guardados.
        Use before downsampling!
        """
        if self.rect_maps is None:
            print("⚠️ Advertencia: Se llamó a rectify() pero no hay mapas cargados!!.")
            if self.info_l!= None and self.info_r!= None:
                cprint("Generate rectification maps","yellow")
                self.generate_rectification_maps()
            else:
                cprint("missing camera infos! Can't rectify","red")
                return
                
        # Desempaquetar mapas
        (ml1, ml2), (mr1, mr2) = self.rect_maps

        # Aplicar remap izquierda
        self.processed_left = cv2.remap(self.processed_left, ml1, ml2, cv2.INTER_LINEAR)

        # Aplicar remap derecha (si es estéreo)
        if self.is_stereo and mr1 is not None:
            self.processed_right = cv2.remap(self.processed_right, mr1, mr2, cv2.INTER_LINEAR)
        
        return self 

    def downsample(self, scale=0.5):
        if scale == 1.0: return self
        
        width = int(self.processed_left.shape[1] * scale)
        height = int(self.processed_left.shape[0] * scale)
        dim = (width, height)
        
        self.processed_left = cv2.resize(self.processed_left, dim, interpolation=cv2.INTER_AREA)
        if self.is_stereo:
            self.processed_right = cv2.resize(self.processed_right, dim, interpolation=cv2.INTER_AREA)
        return self
    
    @staticmethod
    def _custom_gray(img,w_g, w_b, w_r):
        if len(img.shape) < 3: return img
        # Fórmula: 0.7*G + 0.3*B (Ignoramos Rojo)
        gray = (w_g * img[:,:,1] + w_b * img[:,:,0]+w_r*img[:,:,2]).astype(np.uint8)
        return gray

    def convert_to_custom_grayscale(self,w_g=0.7,w_b=0.3,w_r=0.0):
        w_g, w_b, w_r = w_g, w_b, w_r
        
        self.processed_left = self._custom_gray(self.processed_left,w_g, w_b, w_r)
        if self.is_stereo:
            self.processed_right = self._custom_gray(self.processed_right,w_g, w_b, w_r)
        return self
    
    @staticmethod
    def _match_histogram_stats(source, reference):
        m_src, s_src = cv2.meanStdDev(source)
        m_ref, s_ref = cv2.meanStdDev(reference)
        if s_src[0][0] < 1e-3: return source
        ratio = s_ref[0][0] / s_src[0][0]
        res = (source.astype(np.float32) - m_src[0][0]) * ratio + m_ref[0][0]
        return np.clip(res, 0, 255).astype(np.uint8)

    def match_histograms(self,reference="left"):
        """Ajusta el histograma de la DERECHA para que coincida con la IZQUIERDA."""
        if not self.is_stereo:
            return self
        # Default mode uses left_image as reference
        if reference == "left": 
            self.processed_right = self._match_histogram_stats(self.processed_right, self.processed_left)
            
        if reference == "right":
            self.processed_left = self._match_histogram_stats(self.processed_left, self.processed_right)
        return self
    
    def _clahe(self,img):
        if len(img.shape) == 2: # Grayscale
            return self.clahe.apply(img)
        # Color (LAB space is optimal for CLAHE)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = self.clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def apply_clahe(self, clip_limit=3.0, grid_size=(8,8)):
        """
        Applies clahe, for our case grayscale + normal clahe is preferred but it has a 
        Color image option too using lab space (recomended for clahe)
        
        """
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        self.processed_left =self._clahe(self.processed_left)
        if self.is_stereo:
            self.processed_right = self._clahe(self.processed_right)
        return self
    
    def visualize_and_save(self, window_name="Preview", wait_time=0, save_folder=None, frame_id=""):
        """
        Muestra la comparación: Original (Redimensionada) vs Procesada (Real).
        Es mejor bajar la resolución de la original para ver la procesada píxel a píxel 
        tal como la verá el algoritmo estéreo.
        """
        # 1. Obtenemos dimensiones de la imagen DE TRABAJO (la procesada)
        h_proc, w_proc = self.processed_left.shape[:2]
        
        # --- PREPARAR LEFT ---
        # A) Redimensionar Original para que coincida con la procesada
        # Usamos INTER_AREA porque estamos reduciendo, da la mejor calidad sin aliasing
        vis_orig_l = cv2.resize(self.original_left, (w_proc, h_proc), interpolation=cv2.INTER_AREA)
        
        # B) Preparar la procesada para visualización (Deep Copy para no tocar la real)
        vis_proc_l = self.processed_left.copy()
        
        # Normalizar si es float (por seguridad)
        if vis_proc_l.dtype != np.uint8:
            vis_proc_l = cv2.normalize(vis_proc_l, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Convertir a BGR si es Gris (para poder pegar al lado de la original color)
        if len(vis_proc_l.shape) == 2:
            vis_proc_l = cv2.cvtColor(vis_proc_l, cv2.COLOR_GRAY2BGR)

        # C) Concatenar Horizontalmente: [ Original Reducida | Procesada Real ]
        combined_img = np.hstack((vis_orig_l, vis_proc_l))

        # --- PREPARAR RIGHT (Si es Stereo) ---
        if self.is_stereo:
            # A) Original Right
            vis_orig_r = cv2.resize(self.original_right, (w_proc, h_proc), interpolation=cv2.INTER_AREA)
            
            # B) Processed Right
            vis_proc_r = self.processed_right.copy()
            if vis_proc_r.dtype != np.uint8:
                vis_proc_r = cv2.normalize(vis_proc_r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            if len(vis_proc_r.shape) == 2:
                vis_proc_r = cv2.cvtColor(vis_proc_r, cv2.COLOR_GRAY2BGR)
            
            # C) Concatenar Right
            combined_r = np.hstack((vis_orig_r, vis_proc_r))
            
            # D) Juntar todo Verticalmente
            # [ Left  Original | Left  Procesada ]
            # [ Right Original | Right Procesada ]
            combined_img = np.vstack((combined_img, combined_r))

        # 2. Mostrar
        cv2.imshow(window_name, combined_img)
        cv2.waitKey(wait_time)

        # 3. Guardar en disco (Opcional)
        if save_folder is not None:
            import os
            os.makedirs(save_folder, exist_ok=True)
            
            # Guardamos las imágenes ÚTILES para el algoritmo (las procesadas)
            fname_l = f"{frame_id}_left_proc.png"
            cv2.imwrite(os.path.join(save_folder, fname_l), self.processed_left)
            
            if self.is_stereo:
                fname_r = f"{frame_id}_right_proc.png"
                cv2.imwrite(os.path.join(save_folder, fname_r), self.processed_right)
                
            # (Opcional) Guardar también el collage de debug para que lo veas luego
            fname_debug = f"{frame_id}_debug_view.jpg"
            cv2.imwrite(os.path.join(save_folder, fname_debug), combined_img)

        return self
    
    #-----------------------------------------------------------------------------------
    # CUSTOM PROCESSING FOR SARMIENTO DE GAMBOA IMAGES (FIRST CAMPAIGN)
    def apply_deep_sea_process(self, n_clahe=2):
        self.processed_left = self._preprocess_deep_sea_static(self.processed_left, n_clahe)
        if self.is_stereo:
            self.processed_right = self._preprocess_deep_sea_static(self.processed_right, n_clahe)
        return self

    @staticmethod
    def _preprocess_deep_sea_static(image_bgr, n_clahe=2):
        img = image_bgr.copy() # Copia local para cálculo
        img = img.astype(np.float32)
        img[:, :, 2] = img[:, :, 2] * 0.6 
        img = np.clip(img, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for _ in range(n_clahe):
            v_channel = clahe.apply(v_channel)
        hsv[:, :, 2] = v_channel
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)