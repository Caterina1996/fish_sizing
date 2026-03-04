import cv2
import numpy as np
from termcolor import cprint
import os

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
    
    def match_brightness_linear(self, reference="left"):
        """
        Iguala brillo y contraste usando una transformación lineal (Bias y Gain).
        Es mucho más estable para estéreo que el histogram matching completo.
        Formula: I_new = (I - mean_old) * (std_ref / std_old) + mean_ref
        """
        if self.processed_left is None or self.processed_right is None:
            return

        # Definir fuente y referencia
        if reference == "left":
            src = self.processed_right
            ref = self.processed_left
        else:
            src = self.processed_left
            ref = self.processed_right

        # Convertir a LAB para trabajar solo sobre luminancia (Canal L)
        # Esto evita distorsionar los colores, solo iguala la luz
        src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
        ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)
        
        l_src, a_src, b_src = cv2.split(src_lab)
        l_ref, _, _ = cv2.split(ref_lab)

        # Calcular medias y desviaciones estándar
        mean_src, std_src = cv2.meanStdDev(l_src)
        mean_ref, std_ref = cv2.meanStdDev(l_ref)
        
        mean_src = mean_src[0][0]
        std_src = std_src[0][0]
        mean_ref = mean_ref[0][0]
        std_ref = std_ref[0][0]

        # Evitar división por cero
        if std_src < 1e-5: std_src = 1e-5

        # Aplicar transformación lineal
        # gain = std_ref / std_src
        # bias = mean_ref - mean_src * gain
        
        l_res = np.float32(l_src)
        l_res = (l_res - mean_src) * (std_ref / std_src) + mean_ref
        l_res = np.clip(l_res, 0, 255).astype(np.uint8)

        # Reconstruir imagen
        res_lab = cv2.merge((l_res, a_src, b_src))
        result = cv2.cvtColor(res_lab, cv2.COLOR_LAB2BGR)

        # Guardar resultado en la variable de clase correspondiente
        if reference == "left":
            self.processed_right = result
        else:
            self.processed_left = result
    
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
    
    def apply_dehaze(self, omega=0.90, window_size=15, stereo_consistency=True):
        """
        Aplica Dehazing.
        Args:
            stereo_consistency (bool): 
                - True (Recomendado para 3D): Usa la 'A' de la izquierda para ambas. 
                  Garantiza colores iguales.
                - False (Debug/Visual): Calcula 'A' independientemente. 
                  Puede que una quede más brillante que la otra.
        """
        # 1. Procesamos siempre la izquierda primero
        # Obtenemos la imagen procesada y el valor A que calculó
        self.processed_left, A_val_left = self._dehaze_dcp(
            self.processed_left, omega, window_size, known_A=None
        )
        
        # 2. Procesamos la derecha
        if self.is_stereo:
            # Si queremos consistencia, pasamos el A de la izquierda.
            # Si NO queremos (False), pasamos None para que calcule el suyo propio.
            A_to_use = A_val_left if stereo_consistency else None
            
            self.processed_right, _ = self._dehaze_dcp(
                self.processed_right, omega, window_size, known_A=A_to_use
            )
            
        return self
    
    
    def apply_gamma(self, gamma=1.2):
        """Ajusta el brillo no linealmente para revelar detalles en las sombras."""
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        
        self.processed_left = cv2.LUT(self.processed_left, table)
        if self.is_stereo:
            self.processed_right = cv2.LUT(self.processed_right, table)
        return self
    
    def apply_sharpen(self, alpha=1.5):
        """Aumenta la nitidez para que el estéreo (SGBM) tenga 'micro-texturas' a las que agarrarse."""
        blurred_l = cv2.GaussianBlur(self.processed_left, (0, 0), 3)
        self.processed_left = cv2.addWeighted(self.processed_left, alpha, blurred_l, 1 - alpha, 0)
        
        if self.is_stereo:
            blurred_r = cv2.GaussianBlur(self.processed_right, (0, 0), 3)
            self.processed_right = cv2.addWeighted(self.processed_right, alpha, blurred_r, 1 - alpha, 0)
        return self
    
    def apply_bilateral(self, d=5, sigma_color=50, sigma_space=50):
        """Suaviza el ruido del agua manteniendo los bordes del pez intactos."""
        self.processed_left = cv2.bilateralFilter(self.processed_left, d, sigma_color, sigma_space)
        if self.is_stereo:
            self.processed_right = cv2.bilateralFilter(self.processed_right, d, sigma_color, sigma_space)
        return self

    @staticmethod
    def _guided_filter(I, p, r, eps):
        """
        Filtro Guiado para suavizar el mapa de transmisión respetando bordes.
        I: Imagen guía (normalizada 0-1, grayscale)
        p: Imagen a filtrar (mapa de transmisión rudo)
        r: Radio del filtro
        eps: Regularización
        """
        # Calcular medias con filtro de caja (box filter)
        mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
        mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
        
        # Covarianza de (I, p)
        cov_Ip = mean_Ip - mean_I * mean_p
        
        # Varianza de I
        mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
        var_I = mean_II - mean_I * mean_I
        
        # Coeficientes lineal a y b
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        # Medias de a y b
        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
        
        # Resultado refinado
        q = mean_a * I + mean_b
        return q

    @staticmethod
    def _dehaze_dcp(img, omega=0.95, window_size=15,known_A=None):
        """
        Dark Channel Prior MEJORADO con Guided Filter (para eliminar el efecto de bloques cuadrados)
        Acepta un 'known_A' opcional para forzar el color ambiental.
        Retorna (imagen_procesada, A_usado).
        """
        
        if img is None: return None, None
        
        # Trabajar con float64 para precisión
        I = img.astype('float64') / 255.0
        
        # 1. Calcular Dark Channel
        min_channel = np.min(I, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
        dark_channel = cv2.erode(min_channel, kernel)
        
        # 2. Estimar Luz Atmosférica (A)

        if known_A is not None:
            # Si nos dan A, lo usamos (para la cámara derecha)
            A = known_A
        else:
            # Si no, lo calculamos (para la cámara izquierda)
            num_pixels = dark_channel.size
            # Usamos el 0.1% más brillante para estimar la luz ambiental
            num_brightest = int(max(num_pixels * 0.001, 1))
            indices = np.argpartition(dark_channel.ravel(), -num_brightest)[-num_brightest:]
            
            flat_I = I.reshape(num_pixels, 3)
            # Promediamos los candidatos para ser más robustos ante 'pixels quemados'
            A = np.mean(flat_I[indices], axis=0)
            
            # Evitar A demasiado bajo (división por cero) o demasiado alto (saturación)
            # Para agua, a veces conviene forzar A a ser un poco azulado/verdoso si falla,
            # pero el automático suele ir bien.
            A = np.maximum(A, 0.05) 

        # 3. Calcular Transmisión Ruda (t_raw)
        # Normalizamos por A canal a canal
        norm_I = I / A
        min_channel_norm = np.min(norm_I, axis=2)
        dark_channel_norm = cv2.erode(min_channel_norm, kernel)
        
        t_raw = 1 - omega * dark_channel_norm
        
        # 4. REFINAR TRANSMISIÓN (Guided Filter) - AQUÍ SE ARREGLAN LOS CUADRADOS
        # Usamos la versión en gris de la imagen original como guía
        gray_guide = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float64') / 255.0
        
        # Radio grande (ej: 8 veces la ventana) para suavizar bien
        t_refined = ImageProcessor._guided_filter(gray_guide, t_raw, r=window_size*4, eps=1e-3)
        
        # Limitar t para evitar ruido extremo (0.1 es un buen límite inferior)
        t_refined = np.maximum(t_refined, 0.1)
        
        # 5. Recuperar Escena (J)
        t_3c = np.repeat(t_refined[:, :, np.newaxis], 3, axis=2)
        J = (I - A) / t_3c + A
        
        # Clip seguro
        res = np.clip(J * 255, 0, 255).astype(np.uint8)
        return res, A
    
    def run_pipeline(self, pipeline_steps, base_debug_folder=None, frame_id="",visualize=True):
        """
        Ejecuta una secuencia de pasos definida por configuración.
        
        Args:
            pipeline_steps (list): Lista de tuplas [("func_name", {kwargs}, save_debug_bool), ...]
            base_debug_folder (str): Carpeta raíz para guardar los pasos de debug.
            frame_id (str): Identificador del frame para nombrar los archivos.
        """
        cprint(f"   ⚙️ Ejecutando Pipeline...", "cyan")
        
        if base_debug_folder:
            os.makedirs(base_debug_folder, exist_ok=True)

        for i, step in enumerate(pipeline_steps):
            
            func_name, kwargs, save_step = step
            
            # 2. Ejecutar la función
            if hasattr(self, func_name):
                func = getattr(self, func_name)
               
                # Ejecutar processing step
                func(**kwargs)
                
                # 3. Guardar Debug si la tupla lo indica
                if save_step and base_debug_folder:
                    # Creamos un ID único para que salgan ordenados alfabéticamente en la carpeta
                    # Ej: "frame_100_step_01_apply_clahe"
                    step_id = f"{frame_id}_step_{func_name}"
                    
                    # Llamamos a tu función visualize_and_save
                    # Ponemos wait_time=1 para que refresque la ventana pero no bloquee (o 0 si quieres pausar)
                    self.visualize_and_save(
                        window_name=f"Debug: {func_name}", 
                        wait_time=1, 
                        save_folder=base_debug_folder, 
                        frame_id=step_id,
                        visualize=visualize
                    )
                   
            else:
                cprint(f"⚠️ [ERROR]: La función '{func_name}' no existe. Saltando.", "red")
        
        return self.get_processed()
    
    def visualize_and_save(self, window_name="Preview", wait_time=0, save_folder=None, frame_id="",visualize=True):
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

        # Definimos una altura máxima cómoda para ver en monitor (ej. 900px)
        MAX_DISPLAY_HEIGHT = 900  
        
        h_final, w_final = combined_img.shape[:2]
        
        if h_final > MAX_DISPLAY_HEIGHT:
            scale_factor = MAX_DISPLAY_HEIGHT / h_final
            # Calculamos nuevas dimensiones
            new_w = int(w_final * scale_factor)
            new_h = int(h_final * scale_factor)
            # Creamos una copia pequeña SOLO para mostrar con imshow
            img_to_show = cv2.resize(combined_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_to_show = combined_img
        if visualize:
            cv2.imshow(window_name, img_to_show)
            
            if wait_time == 0:
                # Bucle infinito hasta pulsar tecla o cerrar ventana
                while True:
                    key = cv2.waitKey(100) # Chequear cada 100ms
                    
                    # Si se pulsó una tecla válida (distinto de -1) salimos
                    if key != -1:
                        break
                    
                    # Si se cerró la ventana con la X (propiedad WND_PROP_VISIBLE < 1)
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        print("⚠️ Ventana cerrada manualmente. Continuando ejecución...")
                        break
            else:
                # Si hay un tiempo definido (ej. modo vídeo), usamos el wait normal
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
    
    ## CHECK RECTIFICATION:
    # -------------------------------------------------------------------------
    # HERRAMIENTA INTERACTIVA DE VALIDACIÓN DE RECTIFICACIÓN
    # -------------------------------------------------------------------------

    def check_rectification_interactive(self):
        """
        Abre una ventana GUI interactiva para comprobar la rectificación.
        Permite clicar en la imagen izquierda y derecha para medir el error vertical en píxeles.
        Bloquea la ejecución hasta que se pulsa 'q'.
        """
        if not self.is_stereo:
            print("❌ Error: Se necesitan dos imágenes para comprobar rectificación.")
            return

        print("--- MODO COMPROBACIÓN RECTIFICACIÓN ---")
        print("1. Clic en un punto característico en la IZQUIERDA.")
        print("2. Clic en el mismo punto en la DERECHA.")
        print("3. La línea verde debe pasar por ambos.")
        print("Pulsa 'q' para salir.")

        # 1. Preparar la imagen combinada
        # Convertimos a BGR para poder pintar colores (rojo/verde) aunque la imagen sea gris
        img_l = self.processed_left
        img_r = self.processed_right

        if len(img_l.shape) == 2: img_l = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)
        if len(img_r.shape) == 2: img_r = cv2.cvtColor(img_r, cv2.COLOR_GRAY2BGR)

        # Unimos horizontalmente
        self._vis_img = np.hstack((img_l, img_r))
        self._vis_clone = self._vis_img.copy() # Copia limpia para borrar dibujos
        
        # Variables de estado para el callback
        self._pt_left = None
        self._pt_right = None
        self._w_single = img_l.shape[1] # Ancho de una sola imagen

        # 2. Configurar Ventana
        window_name = 'Check Rectification (Press q to exit)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Ajustar tamaño inicial (opcional, para que quepa en pantalla)
        h, w = self._vis_img.shape[:2]
        display_width = 1200
        if w > display_width:
            scale = display_width / w
            cv2.resizeWindow(window_name, display_width, int(h * scale))

        # 3. Asignar Callback
        # Pasamos 'self' implícitamente porque es un método de instancia
        cv2.setMouseCallback(window_name, self._rect_mouse_callback)

        # 4. Bucle de visualización
        while True:
            cv2.imshow(window_name, self._vis_img)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or key == 27: # 'q' o ESC
                break
        
        cv2.destroyWindow(window_name)
        # Limpiamos variables temporales para liberar memoria
        del self._vis_img
        del self._vis_clone

    def _rect_mouse_callback(self, event, x, y, flags, param):
        """Callback interno para manejar los clics del ratón."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Restaurar imagen limpia para borrar líneas anteriores
            self._vis_img = self._vis_clone.copy()

            # Dibujar línea horizontal maestra (Nivel de rectificación)
            # Cruza toda la pantalla a la altura Y del clic
            cv2.line(self._vis_img, (0, y), (self._vis_img.shape[1], y), (0, 255, 0), 1)
            
            # Dibujar punto del clic
            cv2.circle(self._vis_img, (x, y), 5, (0, 0, 255), -1)

            # Lógica de detección de lado
            if x < self._w_single:
                self._pt_left = (x, y)
                # print(f"Click IZQUIERDA: Y={y}")
            else:
                # Guardamos la coordenada relativa a la imagen derecha, 
                # pero para el cálculo de error Y solo nos importa la Y absoluta, que es la misma.
                self._pt_right = (x, y) 
                # print(f"Click DERECHA:   Y={y}")

            # Calcular error si tenemos los dos puntos
            if self._pt_left is not None and self._pt_right is not None:
                # El error es la diferencia de altura (Y)
                y_left = self._pt_left[1]
                y_right = self._pt_right[1]
                diff = abs(y_left - y_right)
                
                msg = f"Error Vertical: {diff} px"
                color_text = (0, 255, 0) # Verde por defecto
                
                if diff > 2: # Tolerancia de 1-2 píxeles es aceptable
                    msg += " (MAL)"
                    color_text = (0, 0, 255) # Rojo
                else:
                    msg += " (OK)"

                # Escribir en pantalla (arriba a la izquierda)
                cv2.putText(self._vis_img, msg, (30, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_text, 3, cv2.LINE_AA)
                
                # Dibujar también el punto anterior para referencia
                if x < self._w_single: # Si acabamos de clicar izq, pintamos el derecho viejo
                     cv2.circle(self._vis_img, self._pt_right, 5, (0, 255, 255), -1)
                else: # Si clicamos der, pintamos el izquierdo viejo
                     cv2.circle(self._vis_img, self._pt_left, 5, (0, 255, 255), -1)

            cv2.imshow('Check Rectification (Press q to exit)', self._vis_img)
    
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