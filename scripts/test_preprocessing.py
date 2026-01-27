#!/usr/bin/env python3.9

import cv2
import argparse
import sys
import os
import numpy as np

from termcolor import cprint
from fish_sizing.utils.image_processor import ImageProcessor



PATH_MAPPINGS = {
    "/home/slimbook/bagfiles": "/home/rosuser/repo/dataset/bagfiles",
    # "/home/slimbook/fish_sizing": "/home/rosuser/repo/src/fish_sizing", # Quitada la barra final para evitar dobles //
    "/home/slimbook/fish_sizing/out": "/home/rosuser/repo/out"
}

USE_DOCKER = True

# CORRECCIÓN: Comas añadidas y typo arreglado en 'right'
TOPICS_DICT = { 
    "left":   "/stereo_ch3/left/image_raw",
    "right":  "/stereo_ch3/right/image_raw",
    "info_l": "/stereo_ch3/left/camera_info",
    "info_r": "/stereo_ch3/right/camera_info"
}

def transform_path2docker(path: str) -> str:
    """
    Transform a path from local computer to docker structure.
    """
    if not USE_DOCKER or path is None:
        return path

    # Recorremos los mapeos
    for host_path, docker_path in PATH_MAPPINGS.items():
        if host_path in path:
            new_path = path.replace(host_path, docker_path)
            cprint(f"🔄 Path mapped: {path} \n   -> {new_path}", "yellow")
            return new_path # Retornamos inmediatamente si encontramos match
            
    # Si llegamos aquí, no hubo cambios
    # cprint(f"No mapping needed for: {path}", "green")
    return path


def main():
    parser = argparse.ArgumentParser(description="Testing the imageprocessing/enhancing class")
    # Rutas por defecto del HOST (Slimbook)
    parser.add_argument("-l", "--left", type=str, 
                        default="/home/slimbook/fish_sizing/out/test_image_pair/1755776074729348982_left_image.jpg", 
                        help="Ruta a la imagen IZQUIERDA")
    parser.add_argument("-r", "--right", type=str,
                        default="/home/slimbook/fish_sizing/out/test_image_pair/1755776074729348982_right_image.jpg", 
                        help="Ruta a la imagen DERECHA (Opcional)")
    parser.add_argument("-o", "--output", type=str, 
                        default="/home/slimbook/fish_sizing/out/image_processing_tests/", 
                        help="Carpeta para guardar resultados")
    
    args = parser.parse_args()
    
    # Transformar rutas
    path_left = transform_path2docker(args.left)
    path_right = transform_path2docker(args.right)
    path_out = transform_path2docker(args.output)

    # 1. Cargar imágenes
    print(f"📸 Cargando Left: {path_left}")
    img_l = cv2.imread(path_left)
    if img_l is None:
        print(f"❌ Error: No se pudo leer la imagen izquierda en: {path_left}")
        sys.exit(1)

    img_r = None
    if path_right:
        print(f"📸 Cargando Right: {path_right}")
        img_r = cv2.imread(path_right)
        # Nota: No salimos con exit si falla la derecha, a lo mejor queremos probar mono
        if img_r is None:
            print(f"⚠️ Aviso: No se pudo leer la imagen derecha en: {path_right}")

    # Creación carpeta output si no existe
    if path_out:
        os.makedirs(path_out, exist_ok=True)

    # 2. Inicializar Procesador
    processor = ImageProcessor(img_l, right_image=img_r)

    print("\n--- INICIANDO TEST VISUAL (Pulsa cualquier tecla para avanzar) ---")

    # =========================================================================
    # TEST 2: PIPELINE "SGBM PREP"
    # =========================================================================
    print("2️⃣  Aplicando 'SGBM Prep'...")
    processor.reset()
    
    processor.rectify \
             .downsample(scale=0.5) \
             .convert_to_custom_grayscale(w_g=0.7, w_b=0.3, w_r=0.0) \
             .match_histograms() \
             .apply_clahe(clip_limit=3.0)
    
    # CORRECCIÓN: Usar path_out (ruta docker), no args.output (ruta host)
    processor.visualize_and_save(
        window_name="SGBM Prep Result", 
        wait_time=0, 
        save_folder=path_out,  # <--- IMPORTANTE
        frame_id="test_sgbm"
    )
    cv2.destroyAllWindows()
    print(f"\n✅ Test finalizado. Resultados guardados en: {path_out}")
    

if __name__ == "__main__":
    main()