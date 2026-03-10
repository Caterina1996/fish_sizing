import numpy as np
import yaml
import os
from tqdm import tqdm

import rosbag
import cv2
from typing import TypedDict, List, Dict, Any
from termcolor import cprint
import logging
from fish_sizing.utils.tools import cprint_and_log

# Para evitar un error con python 3.9...
# Això no és lo seu pero bueno...
class CvBridge:
    """
    Implementación pura en Python de CvBridge.
    Soporta: RGB, BGR, Mono y BAYER (RAW) -> Color.
    """
    def imgmsg_to_cv2(self, img_msg, desired_encoding="passthrough"):
        dtype = np.uint8
        
        # 1. Análisis de canales
        pixel_count = img_msg.width * img_msg.height
        data_len = len(img_msg.data)
        
        # Si el tamaño de datos es el triple que pixeles, es color nativo
        if data_len == pixel_count * 3:
            n_channels = 3
        else:
            n_channels = 1 # Puede ser Grayscale o Bayer RAW
            if "16" in img_msg.encoding:
                dtype = np.uint16

        # 2. Convertir buffer a numpy
        img_buf = np.frombuffer(img_msg.data, dtype=dtype)
        
        # 3. Reshape inicial
        try:
            if n_channels == 3:
                img = img_buf.reshape(img_msg.height, img_msg.width, 3)
            else:
                img = img_buf.reshape(img_msg.height, img_msg.width)
        except ValueError as e:
            # Fallback de emergencia
            cprint_and_log(f"❌ Error de reshape en CvBridge: {e}", "red", level=logging.ERROR)
            return np.zeros((img_msg.height, img_msg.width, 3), dtype=np.uint8)
            
        # 4. LÓGICA DE COLOR Y DEBAYERING
        encoding = img_msg.encoding.lower()

        # CASO A: La imagen YA viene en 3 canales (RGB/BGR)
        if n_channels == 3:
            if desired_encoding == "bgr8" and "rgb" in encoding:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
        # CASO B: La imagen viene en 1 canal (RAW/Bayer) pero queremos Color
        elif n_channels == 1 and desired_encoding == "bgr8":
            if "bayer" in encoding:
                # Necesitamos 'revelar' el RAW (Demosaicing)
                # Mapeo típico de ROS a OpenCV
                if "rggb" in encoding:
                    code = cv2.COLOR_BayerBG2BGR 
                elif "bggr" in encoding:
                    code = cv2.COLOR_BayerRG2BGR
                elif "gbrg" in encoding:
                    code = cv2.COLOR_BayerGR2BGR
                elif "grbg" in encoding:
                    code = cv2.COLOR_BayerGB2BGR
                else:
                    # Default común para muchas cámaras
                    code = cv2.COLOR_BayerBG2BGR 
                
                try:
                    img = cv2.cvtColor(img, code)
                except Exception:
                    cprint_and_log(f"⚠️ Fallo al revelar RAW/Bayer '{encoding}': {e}. Forzando paso a gris.", "yellow", level=logging.WARNING)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) 
            else:
                # If it's a standard mono8 image, explicitly convert it 
                # so it returns a (H, W, 3) shape as requested by "bgr8".
                cprint_and_log(f"⚠️ AVISO: Se recibió imagen de 1 canal '{encoding}' cuando se esperaba color. Convirtiendo a BGR falso.", "yellow", level=logging.WARNING)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img
    
    def imgcompressed_to_cv2(self, img_msg, desired_encoding="bgr8"):
        """
        Versión para CompressedImage que maneja Bayer y Mono.
        """
        # 1. Convertir el buffer de bytes a un array de numpy
        np_arr = np.frombuffer(img_msg.data, np.uint8)
        
        # 2. Decodificar la imagen (OpenCV detecta si es PNG o JPG automáticamente)
        # Usamos IMREAD_ANYCOLOR para mantener la profundidad si fuera necesario
        img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError("❌ No se pudo decodificar la imagen comprimida.")

        # 3. Recuperar el encoding original del campo 'format'
        # Tu script guarda algo como: "rggb8; png compressed rggb8"
        fmt_str = getattr(img_msg, 'format', '').lower()

        # 4. Lógica de Debayering / Conversión de color
        # Si la imagen es de un solo canal (Bayer o Mono)
        if len(img.shape) == 2:
            if "rggb" in fmt_str:
                img = cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)
            elif "bggr" in fmt_str:
                img = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
            elif "gbrg" in fmt_str:
                img = cv2.cvtColor(img, cv2.COLOR_BayerGB2BGR)
            elif "grbg" in fmt_str:
                img = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
            elif "mono" in fmt_str or "8uc1" in fmt_str:
                cprint_and_log(f"⚠️ AVISO: Imagen comprimida es Mono ('{fmt_str}'). Convirtiendo a BGR.", "yellow", level=logging.WARNING)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                # Si no reconoce el bayer, al menos lo pasamos a BGR para que no falle
                cprint_and_log(f"⚠️ AVISO: Formato 1 canal desconocido ('{fmt_str}'). Forzando a BGR.", "yellow", level=logging.WARNING)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
        # 5. Si ya es color (3 canales) pero está en RGB, pasar a BGR para OpenCV
        elif len(img.shape) == 3 and "rgb" in fmt_str:
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img


class StereoTopics(TypedDict):
    left: str
    right: str
    info_l: str
    info_r: str

class BagProcessor:
    def __init__(self, bag_path, topics: StereoTopics):
        self.bag_path = bag_path
        
        # Validate that we receive the required topics:
        required_keys = {'left', 'right', 'info_l', 'info_r'}
        
        if not all(key in topics for key in required_keys):
            msg = f"The 'topics' dict must contain the keys: {required_keys}"
            cprint_and_log(msg, "red", ["bold"], level=logging.CRITICAL)
            raise ValueError(msg)

        # Resolve topics dynamically right at initialization!
        self.topics = self._resolve_all_topics(topics)
        
        self.bridge = CvBridge()
        # self.camera_info = []
        
    def _resolve_all_topics(self, base_topics: StereoTopics):
        """
        Abre el índice del bag una sola vez para resolver si debemos usar 
        topics raw o /compressed, actualizando el diccionario internamente.
        """
        resolved = base_topics.copy()
        
        try:
            # Abrimos el bag aquí UNA SOLA VEZ para leer todos los topics de golpe
            with rosbag.Bag(self.bag_path, 'r') as bag:
                available_topics = bag.get_type_and_topic_info()[1].keys()
                
                for key in ['left', 'right']:
                    base_name = base_topics[key]
                    compressed_name = f"{base_name}/compressed"
                    
                    if base_name in available_topics:
                        resolved[key] = base_name
                    elif compressed_name in available_topics:
                        resolved[key] = compressed_name
                    else:
                        cprint_and_log(f"⚠️ AVISO: Ni '{base_name}' ni '{compressed_name}' encontrados en el bag!", "yellow", level=logging.WARNING)
                        
        except Exception as e:
            cprint_and_log(f"❌ Error abriendo bagfile para resolver topics: {e}", "red", level=logging.ERROR)
            
        return resolved
        
    def get_calibration(self):
        """
        Lee el bagfile solo hasta encontrar los mensajes de calibración.
        Retorna los parámetros intrínsecos/extrínsecos necesarios.
        """
        camera_info = {'left': None, 'right': None}
        # Leemos solo los topics de info
        with rosbag.Bag(self.bag_path, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[self.topics['info_l'], self.topics['info_r']]):
                
                if topic == self.topics['info_l'] and camera_info['left'] is None:
                    camera_info['left'] = msg
                    
                elif topic == self.topics['info_r'] and camera_info['right'] is None:
                    camera_info['right'] = msg
                          
                if camera_info['left'] != None and camera_info['right'] != None:
                    break # Ya tenemos los dos, dejamos de leer
        
        # Aquí podrías procesar los msg para devolver ya las matrices K, D, P, R en numpy
        # O devolver el objeto msg raw si prefieres procesarlo fuera.
        return camera_info
        
    @staticmethod
    def _msg_to_yaml_dict(msg, cam_name="stereo_ch3"):
        """ 
        Converts camera calibration ROS msg to a standard yaml dict
        
        Args:
            msg (camera_info_msg): camera info msg where the params of the camera can be found
            cam_name (str, optional): camera name. Defaults to "stereo_ch3 (lanty)".
            
        """
        yaml_dict= {
            "image_width": msg.width,
            "image_height": msg.height,
            "camera_name": cam_name,
            "camera_matrix": {
                "rows": 3,
                "cols": 3,
                "data": list(msg.K) # Convertimos a lista estándar de Python
            },
            "distortion_model": msg.distortion_model,
            "distortion_coefficients": {
                "rows": 1,
                "cols": len(msg.D),
                "data": list(msg.D)
            },
            "rectification_matrix": {
                "rows": 3,
                "cols": 3,
                "data": list(msg.R)
            },
            "projection_matrix": {
                "rows": 3,
                "cols": 4,
                "data": list(msg.P)
            }
        }
        return yaml_dict
    
    def save_calibration_yaml(self, output_folder):
        """
        Obtiene la calibración del bag y guarda dos archivos YAML (left.yaml, right.yaml)
        en el formato estándar compatible con ROS/OpenCV.
        """
        # 1. Obtenemos los mensajes crudos
        # (Llama a tu propio método get_calibration)
        cam_info_msgs = self.get_calibration()
        
        if cam_info_msgs['left'] is None and cam_info_msgs['right'] is None:
            print("⚠️ No se encontraron mensajes de calibración en el bag.")
            return

        os.makedirs(output_folder, exist_ok=True)

        # 2. Guardar Left
        if cam_info_msgs['left']:
            left_data = self._msg_to_yaml_dict(cam_info_msgs['left'], "stereo_left")
            left_path = os.path.join(output_folder, "left.yaml")
            
            with open(left_path, 'w') as f:
                yaml.dump(left_data, f, default_flow_style=None)
            print(f"✅ Calibración izquierda guardada en: {left_path}")
        else:
            print("⚠️ No se encontró info para cámara IZQUIERDA")

        # 3. Guardar Right
        if cam_info_msgs['right']:
            right_data =  self._msg_to_yaml_dict(cam_info_msgs['right'], "stereo_right")
            right_path = os.path.join(output_folder, "right.yaml")
            
            with open(right_path, 'w') as f:
                yaml.dump(right_data, f, default_flow_style=None)
            print(f"✅ Calibración derecha guardada en: {right_path}")

    def stream_stereo_pairs(self, tolerance_ns=50000000): # 50ms (0.05s) de margen
        """
        Generador robusto: Empareja imágenes aunque los relojes varíen unos milisegundos.
        """
        buffer_left = {}
        buffer_right = {}
        
        target_topics = [self.topics['left'], self.topics['right']]
        print(f"DEBUG: Buscando imágenes en: {target_topics}")
        
        # Contadores para ver qué está pasando
        cnt_l, cnt_r, cnt_match = 0, 0, 0

        with rosbag.Bag(self.bag_path, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=target_topics):
                
                # Usamos el tiempo del header (cuando se capturó), no 't' (cuando se grabó)
                ts = msg.header.stamp.to_nsec()
                
                try:
                    # --- CORRECCIÓN AQUÍ ---
                    # Detectamos si es comprimido mirando el nombre del topic o el tipo de mensaje
                    if "compressed" in topic or hasattr(msg, 'format'):
                        cv_img = self.bridge.imgcompressed_to_cv2(msg, desired_encoding="bgr8")
                    else:
                        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                except Exception as e:
                    cprint_and_log(f"❌ Error decodificando frame en {topic}: {e}", "red", level=logging.ERROR)
                    continue

                # 1. Meter en el buffer correspondiente
                if topic == self.topics['left']:
                    buffer_left[ts] = cv_img
                    cnt_l += 1
                else:
                    buffer_right[ts] = cv_img
                    cnt_r += 1

                match_ts = None
                best_diff = tolerance_ns 
                
                if topic == self.topics['left']:
                    for r_ts in list(buffer_right.keys()):
                        diff = abs(ts - r_ts)
                        if diff < best_diff:
                            best_diff = diff
                            match_ts = r_ts
                    
                    if match_ts is not None:
                        yield ts, cv_img, buffer_right.pop(match_ts)
                        del buffer_left[ts]
                        cnt_match += 1

                else: 
                    for l_ts in list(buffer_left.keys()):
                        diff = abs(ts - l_ts)
                        if diff < best_diff:
                            best_diff = diff
                            match_ts = l_ts
                    
                    if match_ts is not None:
                        yield match_ts, buffer_left.pop(match_ts), cv_img
                        del buffer_right[ts]
                        cnt_match += 1

                if len(buffer_left) > 50:
                    del buffer_left[min(buffer_left.keys())]
                if len(buffer_right) > 50:
                    del buffer_right[min(buffer_right.keys())]

        print(f"\n--- INFORME FINAL DEL BAG ---")
        print(f"Total Izquierda: {cnt_l}")
        print(f"Total Derecha:   {cnt_r}")
        print(f"Pares Unidos:    {cnt_match}")

    def export_images_to_disk(self, output_folder, topic_key='left', processing_func=None,frames_base_name=None):
        """
        Extrae y guarda imágenes de UN tópico específico.
        
        Args:
            output_folder (str): Ruta donde guardar.
            topic_key (str): 'left' o 'right'.
            processing_func (callable, optional): Función que recibe una imagen BGR y devuelve una procesada.
        """
        target_topic = self.topics.get(topic_key)
        if not target_topic:
            print(f"Error: Tópico {topic_key} no definido.")
            return

        os.makedirs(output_folder, exist_ok=True)
        cprint_and_log(f"Exporting {topic_key} ({target_topic}) to {output_folder}...", "cyan")

        # Abrimos el bag
        with rosbag.Bag(self.bag_path, 'r') as bag:
            # Contamos mensajes para la barra de progreso (opcional)
            n_msgs = bag.get_message_count(target_topic)
            
            frame_counter = 0
            
            for _, msg, t in tqdm(bag.read_messages(topics=[target_topic]), total=n_msgs):
                # 1. Conversión ROS -> OpenCV (BGR)
                if "compressed" in target_topic:
                    cprint_and_log("Processing compressed image...", "magenta", level=logging.DEBUG)
                    cv_image = self.bridge.imgcompressed_to_cv2(msg)
                else:
                    cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                
                # 2. Preprocesado (Solo si se pasa una función)
                if processing_func is not None:
                    cv_image = processing_func(cv_image)
                
                # 3. Guardado
                # Opción A: Nombre con timestamp (bueno para sincronizar luego)
                # timestamp = str(t.to_nsec())
                # filename = f"{timestamp}.png"
                
                # Opción B: Nombre secuencial (frame_001.png)
                filename = f"{frames_base_name}_fr_{frame_counter:05d}.png"
                
                save_path = os.path.join(output_folder, filename)
                cv2.imwrite(save_path, cv_image)
                
                frame_counter += 1
                    

        print(f"✅ Guardadas {frame_counter} imágenes en {output_folder}")