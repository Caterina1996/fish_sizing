#!/usr/bin/env python3.9
import argparse
import os
import sys
import cv2
import numpy as np
from termcolor import cprint


from fish_sizing.utils.bag_processor import BagProcessor
from fish_sizing.utils.image_processor import ImageProcessor


# --- CONFIGURACIÓN ---
PATH_MAPPINGS = {
    "/home/slimbook/bagfiles": "/home/rosuser/dataset/bagfiles",
    "/home/slimbook/fish_sizing/out": "/home/rosuser/repo/out",
    # "/home/slimbook/fish_sizing": "/home/rosuser/repo/src/fish_sizing"
}

USE_DOCKER = True

TOPICS_DICT = { 
    "left":   "/stereo_ch3/left/image_raw",
    "right":  "/stereo_ch3/right/image_raw", 
    "info_l": "/stereo_ch3/left/camera_info",
    "info_r": "/stereo_ch3/right/camera_info"
}

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



# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Script para procesar bagfiles y guardar imágenes filtradas.")
    
    parser.add_argument("--bag_file", "-bg", type=str, 
                        default="/home/slimbook/bagfiles/LIMA/2025/2025_08_21/selec2/13_05_55/stereo_camera_images_2025-08-21-13-05-55_0.bag",
                        help="Ruta al bagfile")
    
    parser.add_argument("--out_path", "-out", type=str, 
                        default="/home/slimbook/fish_sizing/out/test_export",
                        help="Carpeta donde se guardarán las imágenes procesadas")
    
    parser.add_argument("--topic", type=str, default="left", choices=["left", "right"],
                        help="Qué cámara exportar")
    
    args = parser.parse_args()
    
    # 1. Transformar rutas para Docker
    bag_file = transform_path2docker(args.bag_file)
    out_path = transform_path2docker(args.out_path)

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

    # 5. Bucle de Procesamiento (Stereo Stream)
    cprint(f"🚀 Iniciando procesamiento y exportación a: {out_path}", "cyan")
    
    count = 0
    # Usamos stream_stereo_pairs porque necesitamos AMBAS imágenes para rectificar
    for timestamp, img_l_raw, img_r_raw in bag_proc.stream_stereo_pairs():
          
        # A) Cargar el par nuevo en el procesador
        # (Asegúrate de haber corregido el typo 'selfleft' en image_processor.py)
        img_proc.set_image_pair(img_l_raw, img_r_raw)
        
        # B) Pipeline de Procesamiento
        # 1. Rectificar (Geometría)
        img_proc.rectify()
        
        # 2. Reducir tamaño (Opcional, pero recomendado)
        img_proc.downsample(0.5)
        img_proc.apply_dehaze(omega=0.85, window_size=15,stereo_consistency=True) #-> Revisar esto xq ahora mismo no va be/no interesa
        
        # # Opción SGBM Prep (Grayscale + CLAHE - Mejor para Stereo Matcher):
        # img_proc.convert_to_custom_grayscale(w_g=0.7, w_b=0.3, w_r=0.0)
        img_proc.match_histograms(reference="left") # Igualar brillos
        # img_proc.apply_clahe(clip_limit=3.0)
        
        # C) Obtener resultados
        processed_l, processed_r = img_proc.get_processed()
        
        if count == 0: # Ejemplo: Verificar solo el primer frame
            cprint("🛠️ Abriendo verificador de rectificación...", "yellow")
            img_proc.check_rectification_interactive()
        
        # D) Guardar downsampled
        fname = f"{timestamp}"
        cv2.imwrite(os.path.join(out_path, fname+"_left.png"), processed_l)
        cv2.imwrite(os.path.join(out_path, fname+"_right.png"), processed_r)
        
        count += 1
        print(f"Procesado frame par: {count}", end='\r')

    cprint(f"\n✅ Terminado. {count} pares guardados en {out_path}", "green")

if __name__ == "__main__":
    main()