#!/usr/bin/env python3.9
import argparse
import os
import sys
import cv2
import numpy as np
from termcolor import cprint
from ultralytics import YOLO
from pathlib import Path
from fish_sizing.utils.bag_processor import BagProcessor
from fish_sizing.utils.image_processor import ImageProcessor


# --- CONFIGURACIÓN ---
PATH_MAPPINGS = {
    "/home/slimbook/bagfiles": "/home/rosuser/dataset/bagfiles",
    "/home/slimbook/fish_sizing/out": "/home/rosuser/repo/out",
    "/home/slimbook/models": "/home/rosuser/dataset/models/",
    "home/slimbook/fish_sizing/config" :"/home/rosuser/repo/config/"
}

USE_DOCKER = True

TOPICS_DICT = { 
    "left":   "/stereo_ch3/left/image_raw",
    "right":  "/stereo_ch3/right/image_raw", 
    "info_l": "/stereo_ch3/left/camera_info",
    "info_r": "/stereo_ch3/right/camera_info"
}

# TOPICS_DICT = { 
#     "left":   "/stereo_ch3/left/image_raw/compressed",
#     "right":  "/stereo_ch3/right/image_raw/compressed", 
#     "info_l": "/stereo_ch3/left/camera_info",
#     "info_r": "/stereo_ch3/right/camera_info"
# }




BAGFILE_PATH="//home/slimbook/bagfiles/peixos_morts_piscina_v3/2025_05_08/11_14_52/stereo_camera_images_2025-05-08-11-14-53_0.bag"
# BAGFILE_PATH="//home/slimbook/bagfiles/LIMA/2025/2025_08_21/test_comprsesion/13_34_24/stereo_camera_images_2025-08-21-13-34-25_0.bag"
lanty = "L1"

extract = True
infer = True
rename = True

MODEL_PATH = "/home/slimbook/models/yv11l/ylarge_d18_poolv2r_lantytr_nocturnes/weights/best.pt"
# OUT_PATH = "/home/slimbook/fish_sizing/out/test_export/2025-05-08-11-18-25_1/"
OUT_PATH = "/home/slimbook/fish_sizing/out/overfitting_dataset/"

# --- FUNCIONES AUXILIARES ---

def transform_path2docker(path: str) -> str:
    """Transform a path from local computer to docker structure."""
    if not USE_DOCKER or path is None:
        return path

    for host_path, docker_path in PATH_MAPPINGS.items():
        if host_path in path:
            new_path = path.replace(host_path, docker_path)
            cprint(f"🔄 Path mapped: {path} \n   -> {new_path}", "yellow")
            return new_path 
            
    return path

def decode_compressed_image(self, msg):
    np_arr = np.frombuffer(msg.data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise RuntimeError("❌ Error decodificando imagen comprimida")

    return img


# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Script para procesar bagfiles y guardar imágenes filtradas.")
        
    parser.add_argument("--bag_file", "-bg", type=str, help="Ruta al bagfile",
                        default=BAGFILE_PATH)

    parser.add_argument("--out_path", "-out", type=str, help="Carpeta donde se guardarán las imágenes procesadas",
                        default=OUT_PATH)
    
    parser.add_argument("--topic", type=str, default="left", choices=["left", "right"],help="Qué cámara exportar")
    
    parser.add_argument("--model_path", type=str,  help="Path to the detection AI model",
                        default=MODEL_PATH)
    
    args = parser.parse_args()
    
    
    # 1. Transformar rutas para Docker
    bag_file = transform_path2docker(args.bag_file)
    out_path = transform_path2docker(args.out_path)
    model_path = transform_path2docker(args.model_path)

    if not os.path.exists(bag_file):
        cprint(f"❌ Error: El archivo bag no existe: {bag_file}", "red")
        sys.exit(1)

    # 2. Inicializar BagProcessor
    cprint(f"📂 Leyendo bag: {bag_file}", "cyan")
    bag_proc = BagProcessor(bag_file, TOPICS_DICT)

    # 3. Obtener Calibración (Necesaria para rectificar)
    cprint("🔍 Buscando mensajes de calibración...", "yellow")
    camera_info = bag_proc.get_calibration()
    
    if camera_info['left'] is None or camera_info['right'] is None:
        cprint("❌ Error: No se encontró info de calibración en el bag. No se puede rectificar.", "red")
        sys.exit(1)
        
    cprint("✅ Calibración encontrada.", "green")
    
    bag_proc.save_calibration_yaml(out_path)
    
    cprint("✅ Calibración guardada.", "green")

    # 4. Inicializar ImageProcessor (con la info de calibración)
    img_proc = ImageProcessor(
        info_l=camera_info['left'], 
        info_r=camera_info['right']
    )

    # Definimos una función rápida (wrapper) para pasarle al exportador
    # Esta función recibirá la imagen BGR que sale del bridge
    def my_preprocessing(img_bgr):
        if img_bgr is None:
            return None
            
        # 1. Cargar la imagen en el procesador (esto llena self.processed_left)
        # Si solo usas una cámara, puedes pasar la misma dos veces o configurar stereo=False
        img_proc.set_image_pair(img_bgr, img_bgr)
        
        # 2. Aplicar el algoritmo (ahora sí usará los valores por defecto de omega, etc.)
        img_proc.apply_dehaze()
        
        # 3. Retornar el resultado
        return img_proc.processed_left

    # 5. Bucle de Procesamiento (Stereo Stream)
    cprint(f"🚀 Iniciando procesamiento y exportación a: {out_path}", "cyan")
    
    # Cargar Modelo
    cprint(f"🧠 Cargando modelo para inferencia: {model_path}", "cyan")
    model = YOLO(model_path)
    
    count = 0
        
    p = Path(bag_file)

    # 2. Extraer partes
    date_folder = p.parent.parent.name  # "2025_08_21"
    time_folder = p.parent.name         # "13_33_03"
    file_index = p.stem.split('_')[-1]  # p.stem es el nombre sin .bag -> coge el último trozo tras el "_"

    # 3. Juntar
    bag_id = f"{lanty}_{date_folder}-{time_folder}-{file_index}"
    print(bag_id)
    
    if extract:
        # bag_proc.export_images_to_disk(out_path, topic_key="left",frames_base_name=bag_id,processing_func=my_preprocessing)
        # if count % 20 ==0:
            
        #     bag_proc.export_images_to_disk(out_path, topic_key="left",frames_base_name=bag_id)
            
        # count += 1


        # Usamos stream_stereo_pairs porque necesitamos AMBAS imágenes para rectificar
        for timestamp, img_l_raw, img_r_raw in bag_proc.stream_stereo_pairs():
            
            # A) Cargar el par nuevo en el procesador
            # (Asegúrate de haber corregido el typo 'selfleft' en image_processor.py)
            img_proc.set_image_pair(img_l_raw, img_r_raw)
                
            # 2. Reducir tamaño (Opcional, pero recomendado)
            img_proc.downsample(0.5)
            
            # C) Obtener resultados
            processed_l, processed_r = img_proc.get_processed()
                    
            # D) Guardar downsampled
            # fname = f"{timestamp}"
            fname= bag_id+"_f_"+str(count)
            if count % 20 ==0:
                cv2.imwrite(os.path.join(out_path, fname+"_left.png"), processed_l)
                # cv2.imwrite(os.path.join(out_path, fname+"_right.png"), processed_r)
                
            count += 1
            print(f"Procesado frame par: {count}", end='\r')

        cprint(f"\n✅ Terminado. {count} pares guardados en {out_path}", "green")
    
    if infer:
        
        results = model.predict(out_path, 
                conf=0.5, 
                retina_masks=True, 
                line_width=1,
                batch=1, 
                device='cuda', 
                project = out_path,
                name = "inferred",
                half=True,
                agnostic_nms=True,
                show_labels=True,
                save=True, 
                save_txt = True,
                save_conf = False,
                augment=False,
                imgsz=1280, 
                max_det=250,
                boxes=True, 
                exist_ok=True,
                verbose=False)
        
    if rename:

        renamed_count = 0
        inferred_path = Path(out_path) / "inferred"
    
        for file_path in inferred_path.glob("*.jpg"):
            if "_inferred" not in file_path.stem:
                new_file_path = file_path.with_name(
                    f"{file_path.stem}_inferred{file_path.suffix}"
                )
                file_path.rename(new_file_path)
                renamed_count += 1

                    
        cprint(f"✅ Renombrado completado. {renamed_count} archivos actualizados.", "green")
        cprint(f"📁 Resultados finales en: {inferred_path}", "green")

if __name__ == "__main__":
    main()