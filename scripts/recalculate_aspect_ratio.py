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
# IMPORTANTE: Necesario para que pickle pueda reconstruir los objetos correctamente
from fish_sizing.detection.fish2D import Fish2D, FrameScene 

PATH_MAPPINGS = {
    "/home/slimbook/bagfiles": "/home/rosuser/dataset/bagfiles",
    "/home/slimbook/fish_sizing/out": "/home/rosuser/repo/out",
    "/home/slimbook/models": "/home/rosuser/dataset/models/",
    "/home/slimbook/fish_sizing/config" :"/home/rosuser/repo/config/",
    "/media/slimbook/easystore": "/home/rosuser/easystore"
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


def process_single_folder(folder_path):
    """
    Procesa una única carpeta: busca el CSV, calcula el OBB desde los PKL,
    actualiza la columna de pandas y guarda el nuevo CSV directamente.
    """
    folder_name = os.path.basename(folder_path)
    
    # 1. Buscar el CSV original
    csv_path = os.path.join(folder_path, "results_articles_basic", "results", "all_fish_info_raw.csv")
    if not os.path.exists(csv_path):
        cprint(f"  ⏭️ Saltando '{folder_name}': No se encontró el CSV en results_articles_basic/results/", "yellow")
        return False

    # 2. Buscar los archivos .pkl en esa misma carpeta de artículo
    pkl_files = natsorted(glob.glob(os.path.join(folder_path, "results_articles_basic", "*.pkl")))
    if not pkl_files:
        cprint(f"  ⏭️ Saltando '{folder_name}': No hay archivos .pkl", "yellow")
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
    out_dir = os.path.join(folder_path, "corrected_results")
    os.makedirs(out_dir, exist_ok=True) # Creamos las carpetas si no existen
    
    out_csv_path = os.path.join(out_dir, "all_fish_info_raw.csv")
    df_raw.to_csv(out_csv_path, index=False)
    
    cprint(f"  💾 Guardado limpio con Pandas en: {out_csv_path}", "blue")
    return True

def main():
    parser = argparse.ArgumentParser(description="Wrapper para corregir el Aspect Ratio en múltiples carpetas de un dataset.")
    parser.add_argument("--parent_dir", "-p", type=str, default="/media/slimbook/easystore/results_fish_sizing/seleccio_article/2025_05_08/1_peix", help="Carpeta padre que contiene todas las subcarpetas (ej. /dataset/bagfiles)")
    
    args = parser.parse_args()
    
    # Aplicar la conversión de rutas de Docker
    args.parent_dir = transform_path2docker(args.parent_dir)
    
    if not os.path.exists(args.parent_dir):
        cprint(f"❌ Error: La carpeta padre {args.parent_dir} no existe.", "red")
        sys.exit(1)

    cprint(f"🔍 Buscando carpetas con 'results_articles_basic' en: {args.parent_dir}\n", "magenta", attrs=["bold"])

    # El doble asterisco (**) significa "cualquier cantidad de carpetas intermedias"
    patron_busqueda = os.path.join(args.parent_dir, "**", "results_articles_basic")
    
    # IMPORTANTE: Hay que añadir recursive=True para que el ** funcione
    rutas_encontradas = glob.glob(patron_busqueda, recursive=True)
    
    # Nos quedamos con el "dirname" (la carpeta del pez) de los que coincidan
    subcarpetas = [os.path.dirname(ruta) for ruta in rutas_encontradas if os.path.isdir(ruta)]
    subcarpetas = natsorted(subcarpetas)
    
    if not subcarpetas:
        cprint(f"⚠️ No se encontró ninguna subcarpeta que contenga 'results_articles_basic'.", "yellow")
        sys.exit(0)
        
    cprint(f"🎯 Se han encontrado {len(subcarpetas)} carpetas válidas para procesar.\n", "cyan")
    
    carpetas_procesadas = 0
    
    for folder in subcarpetas:
        exito = process_single_folder(folder)
        if exito:
            carpetas_procesadas += 1

    cprint(f"\n🎉 ¡Proceso por lotes finalizado! Se actualizaron {carpetas_procesadas} carpetas con éxito.", "green", attrs=["bold"])

if __name__ == "__main__":
    main()