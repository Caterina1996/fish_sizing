#!/usr/bin/env python3.9
import argparse
import os
import glob
import pickle
import numpy as np
import pandas as pd
import cv2 
from termcolor import cprint, colored
from natsort import natsorted
import sys

from fish_sizing.detection.fish2D import Fish2D, FrameScene 

PATH_MAPPINGS = {
    "/home/slimbook/bagfiles": "/home/rosuser/dataset/bagfiles",
    "/home/slimbook/fish_sizing/out": "/home/rosuser/repo/out",
    "/home/slimbook/models": "/home/rosuser/dataset/models/",
    "/home/slimbook/fish_sizing/config" :"/home/rosuser/repo/config/",
    "/media/slimbook/easystore": "/home/rosuser/easystore",
    "/media/slimbook/easystore2": "/home/rosuser/easystore2",
    "/media/slimbook/easystore1": "/home/rosuser/easystore1"
}

USE_DOCKER = True

def transform_path2docker(path: str) -> str:
    """Convierte rutas de la máquina host a rutas dentro del contenedor Docker."""
    if not USE_DOCKER or path is None:
        return path
    for host_path, docker_path in PATH_MAPPINGS.items():
        if host_path in path:
            new_path = path.replace(host_path, docker_path)
            cprint(f"🔄 Path mapped: {path} \n   -> {new_path}", "yellow")
            return new_path 
    return path


def get_oriented_aspect_ratio(mask):
    """
    Calcula el Aspect Ratio real basándose en un Bounding Box Rotado (OBB)
    ajustado al contorno de la máscara del pez.
    """
    if mask is None or np.sum(mask) == 0:
        return 0.0
        
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0
        
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    (center_x, center_y), (width, height), angle = rect
    
    if min(width, height) > 0:
        return max(width, height) / min(width, height)
    return 0.0


def process_single_folder(folder_path, results_dir):
    """
    Procesa una única carpeta basándose en el parámetro results_dir dinámico.
    """
    folder_name = os.path.basename(folder_path)
    
    # 1. Buscar el CSV original usando la ruta relativa pasada por parámetro
    csv_path = os.path.join(folder_path, results_dir, "all_fish_info_raw.csv")
    if not os.path.exists(csv_path):
        cprint(f"  ⏭️ Saltando '{folder_name}': No se encontró el CSV en {results_dir}/", "yellow")
        return False

    # 2. Búsqueda inteligente de PKLs (Busca en varios niveles cercanos al CSV)
    base_results_path = os.path.dirname(csv_path)       # ej: .../results_article_basic/results
    parent_experiment = os.path.dirname(base_results_path) # ej: .../results_article_basic
    
    pkl_files = natsorted(glob.glob(os.path.join(parent_experiment, "*.pkl")))
    if not pkl_files:
        pkl_files = natsorted(glob.glob(os.path.join(base_results_path, "*.pkl")))
        if not pkl_files:
            pkl_files = natsorted(glob.glob(os.path.join(folder_path, "*.pkl")))
            if not pkl_files:
                cprint(f"  ⏭️ Saltando '{folder_name}': No hay archivos .pkl cerca de {results_dir}", "yellow")
                return False

    cprint(f"\n🐟 Procesando: {folder_name} ({len(pkl_files)} frames)", "cyan", attrs=["bold"])
    
    # 3. Leer CSV
    df_raw = pd.read_csv(csv_path)

    # 4. Actualizar Aspect Ratios desde los PKL
    cambios = 0
    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as f:
            frame_scene = pickle.load(f)
            
        frame_id_str = str(frame_scene.frame_name)

        for fish in frame_scene.fish_list:
            if fish.mask is None:
                continue
                
            new_ar = get_oriented_aspect_ratio(fish.mask)
            
            # Buscar en el dataframe y actualizar
            mask_df = (df_raw['frame_id'].astype(str) == frame_id_str) & (df_raw['track_id'].astype(str) == str(fish.track_id))
            if mask_df.any():
                df_raw.loc[mask_df, 'aspect_ratio'] = new_ar
                cambios += 1

    cprint(f"  ✅ {cambios} Aspect Ratios re-calculados usando OBB.", "green")

    # 5. Actualizar la regla fish_3d_ok
    if 'fish_3d_ok' in df_raw.columns:
        df_raw['fish_3d_ok'] = (
            (df_raw['aspect_ratio'] >= 1.8) & 
            (df_raw['is_3D_complete'] == True) & 
            (df_raw['in_image_borders'] == False) & 
            (df_raw['does_overlap'] == False)
        )

    # 6. GUARDAR DIRECTAMENTE EL CSV MODIFICADO
    # Lo guarda en la misma carpeta padre del 'results', pero llamado 'corrected_results'
    out_dir = os.path.join(parent_experiment, "corrected_results")
    os.makedirs(out_dir, exist_ok=True) 
    
    out_csv_path = os.path.join(out_dir, "all_fish_info_raw.csv")
    df_raw.to_csv(out_csv_path, index=False)
    
    cprint(f"  💾 Guardado limpio con Pandas en: {out_csv_path}", "blue")
    return True


def main():
    parser = argparse.ArgumentParser(description="Wrapper para corregir el Aspect Ratio en múltiples carpetas de un dataset.")
    parser.add_argument("--parent_dir", "-p", type=str, default="/media/slimbook/easystore1/results_fish_sizing/seleccio_article/lanty1/2025_08_21/1_peix/", help="Carpeta padre que contiene todas las subcarpetas")
    parser.add_argument("--results_dir", "-r", type=str, default="results", help="Subcarpeta donde se encuentra el CSV (ej: 'results' o 'results_article_basic_nou/results')")
    
    args = parser.parse_args()
    
    # Aplicar la conversión de rutas de Docker
    args.parent_dir = transform_path2docker(args.parent_dir)
    
    if not os.path.exists(args.parent_dir):
        cprint(f"❌ Error: La carpeta padre {args.parent_dir} no existe.", "red")
        sys.exit(1)

    cprint(f"🔍 Búsqueda rápida de 'original_images' en: {args.parent_dir}\n", "magenta", attrs=["bold"])


    rutas_encontradas = []
    
    for root, dirs, files in os.walk(args.parent_dir):
        # 1. PODAR EL ÁRBOL: Eliminamos de la lista de carpetas a visitar cualquiera que contenga "frame"
        # Esto evita que el script pierda tiempo entrando en miles de carpetas de imágenes
        dirs[:] = [d for d in dirs if "frame" not in d.lower()]
        
        # 2. COMPROBAR SI HEMOS LLEGADO AL ANCLA
        if "original_images" in dirs:
            rutas_encontradas.append(os.path.join(root, "original_images"))
            # Opcional: si sabemos que dentro de original_images no hay más original_images, no entramos
            dirs.remove("original_images")

    # Si encontramos .../12_07_31/0/original_images, nos quedamos con la carpeta padre (.../12_07_31/0)
    subcarpetas = [os.path.dirname(ruta) for ruta in rutas_encontradas]
    subcarpetas = natsorted(list(set(subcarpetas)))
    
    if not subcarpetas:
        cprint(f"⚠️ No se encontró ninguna subcarpeta que contenga 'original_images'.", "yellow")
        sys.exit(0)
        
    cprint(f"🎯 Se han encontrado {len(subcarpetas)} carpetas válidas para procesar.\n", "cyan")
    
    carpetas_procesadas = 0
    
    for folder in subcarpetas:
        exito = process_single_folder(folder, args.results_dir)
        if exito:
            carpetas_procesadas += 1

    cprint(f"\n🎉 ¡Proceso por lotes finalizado! Se actualizaron {carpetas_procesadas} carpetas con éxito.", "green", attrs=["bold"])
    
    print("\n[INFO] RUTAS DETECTADAS COMO ANCLAS:")
    for ruta in rutas_encontradas:
        print(f" - {ruta}")
    print("-" * 50)

if __name__ == "__main__":
    main()