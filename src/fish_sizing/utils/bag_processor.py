import numpy as np
import yaml
import os
from tqdm import tqdm

import rosbag
# from cv_bridge import CvBridge
import cv2
from typing import TypedDict, List, Dict, Any


# Para evitar un error con python 3.9...
# Això no és lo seu pero bueno...
class CvBridge:
    """
    Versión 'falsa' de CvBridge escrita en Python puro.
    Permite leer imágenes de ROS en Python 3.9 sin necesitar la librería compilada de C++.
    """
    def imgmsg_to_cv2(self, img_msg, desired_encoding="passthrough"):
        dtype = np.uint8
        n_channels = 1
        
        # 1. Detectar canales según el nombre del encoding ROS
        if "8UC1" in img_msg.encoding or "mono8" in img_msg.encoding:
            n_channels = 1
        elif "8UC3" in img_msg.encoding or "bgr8" in img_msg.encoding or "rgb8" in img_msg.encoding:
            n_channels = 3
        
        # 2. Convertir los bytes crudos a un array de NumPy
        # Esto es lo que hacía C++ internamente, pero numpy lo hace muy rápido también
        img_buf = np.frombuffer(img_msg.data, dtype=dtype)
        
        # 3. Darle forma (Alto, Ancho, Canales)
        try:
            img = img_buf.reshape(img_msg.height, img_msg.width, n_channels)
        except ValueError:
            # Fallback por si acaso viene plano
            img = img_buf.reshape(img_msg.height, img_msg.width, -1)
            
        # 4. Ajustar orden de colores (ROS suele usar RGB, OpenCV usa BGR)
        if desired_encoding == "bgr8" and "rgb8" in img_msg.encoding:
            img = img[:, :, ::-1] # Invierte el orden de canales
            
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
            raise ValueError(f"El diccionario 'topics' debe contener las claves: {required_keys}")
            
        self.topics = topics
        self.bridge = CvBridge()
        # self.camera_info = []
        
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
                    cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                except Exception as e:
                    continue

                # 1. Meter en el buffer correspondiente
                if topic == self.topics['left']:
                    buffer_left[ts] = cv_img
                    cnt_l += 1
                else:
                    buffer_right[ts] = cv_img
                    cnt_r += 1

                # 2. Intentar buscar pareja en el OTRO buffer
                # (No buscamos exactitud, buscamos al "vecino más cercano" dentro de la tolerancia)
                
                match_ts = None
                best_diff = tolerance_ns # Empezamos con el máximo permitido
                
                if topic == self.topics['left']:
                    # Acaba de llegar L, buscamos en R
                    for r_ts in list(buffer_right.keys()):
                        diff = abs(ts - r_ts)
                        if diff < best_diff:
                            best_diff = diff
                            match_ts = r_ts
                    
                    if match_ts is not None:
                        # ¡Encontrado! Devolvemos el par
                        yield ts, cv_img, buffer_right.pop(match_ts)
                        del buffer_left[ts] # Ya no la necesitamos
                        cnt_match += 1

                else: 
                    # Acaba de llegar R, buscamos en L
                    for l_ts in list(buffer_left.keys()):
                        diff = abs(ts - l_ts)
                        if diff < best_diff:
                            best_diff = diff
                            match_ts = l_ts
                    
                    if match_ts is not None:
                        # ¡Encontrado!
                        yield match_ts, buffer_left.pop(match_ts), cv_img
                        del buffer_right[ts]
                        cnt_match += 1

                # 3. Limpieza de seguridad (para no llenar la RAM si una cámara muere)
                if len(buffer_left) > 50:
                    del buffer_left[min(buffer_left.keys())] # Borrar la más vieja
                if len(buffer_right) > 50:
                    del buffer_right[min(buffer_right.keys())]

        print(f"\n--- INFORME FINAL DEL BAG ---")
        print(f"Total Izquierda: {cnt_l}")
        print(f"Total Derecha:   {cnt_r}")
        print(f"Pares Unidos:    {cnt_match}")
        if cnt_match == 0:
            print("❌ ERROR CRÍTICO: No se han podido emparejar. Revisa los nombres de los topics o aumenta 'tolerance_ns'.")

    def export_images_to_disk(self, output_folder, topic_key='left', processing_func=None):
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
        print(f"Exportando {topic_key} ({target_topic}) a {output_folder}...")

        # Abrimos el bag
        with rosbag.Bag(self.bag_path, 'r') as bag:
            # Contamos mensajes para la barra de progreso (opcional)
            n_msgs = bag.get_message_count(target_topic)
            
            frame_counter = 0
            
            for _, msg, t in tqdm(bag.read_messages(topics=[target_topic]), total=n_msgs):
                try:
                    # 1. Conversión ROS -> OpenCV (BGR)
                    cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    
                    # 2. Preprocesado (Solo si se pasa una función)
                    if processing_func is not None:
                        cv_image = processing_func(cv_image)
                    
                    # 3. Guardado
                    # Opción A: Nombre con timestamp (bueno para sincronizar luego)
                    timestamp = str(t.to_nsec())
                    filename = f"{timestamp}.png"
                    
                    # Opción B: Nombre secuencial (frame_001.png)
                    # filename = f"frame_{frame_counter:05d}.png"
                    
                    save_path = os.path.join(output_folder, filename)
                    cv2.imwrite(save_path, cv_image)
                    
                    frame_counter += 1
                    
                except Exception as e:
                    print(f"Error frame {frame_counter}: {e}")

        print(f"✅ Guardadas {frame_counter} imágenes en {output_folder}")