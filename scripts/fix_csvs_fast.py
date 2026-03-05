#!/usr/bin/env python3.9
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from termcolor import cprint


from fish_sizing.detection.fish2D import Fish2D, FrameScene

def filter_outliers_per_track(df_input, gt_val=None):
    """Función de filtrado inteligente y generación de resumen."""
    resume_list = []
    
    for track_id, track_data in df_input.groupby('track_id'):
        num_frames = len(track_data)
        if num_frames < 3: continue

        sorted_lengths = track_data['filtered_length'].sort_values(ascending=False).tolist()
        median_val = np.median(sorted_lengths)
        valid_max = sorted_lengths[0] 
        
        if len(sorted_lengths) >= 5:
            while len(sorted_lengths) > 2:
                current_max = sorted_lengths[0]
                next_max = sorted_lengths[1]
                current_median = np.median(sorted_lengths) 
                if current_max > (current_median * 2.0):
                    sorted_lengths.pop(0)
                    continue
                if (current_max - next_max) / next_max < 0.05: 
                    valid_max = current_max 
                    break
                else:
                    sorted_lengths.pop(0)
            if len(sorted_lengths) <= 2: valid_max = sorted_lengths[0]

        if num_frames < 20: 
            representative_length = mean_top_20 = valid_max  
        else:
            n_top = max(1, int(len(sorted_lengths) * 0.2))
            top_lengths = sorted_lengths[:n_top]
            representative_length = mean_top_20 = sum(top_lengths) / len(top_lengths)

        abs_err = rel_err = None
        if gt_val is not None and gt_val > 0:
            gt_m = gt_val / 100.0
            abs_err = abs(representative_length - gt_m) * 100 
            rel_err = (abs_err / gt_val) * 100 
            
        stats = {
            'track_id': track_id,
            'class_name': track_data['class_name'].iloc[0],
            'n_frames_validos': num_frames,
            'does_overlap': track_data['does_overlap'].any(),       
            'in_image_borders': track_data['in_image_borders'].any(), 
            'max_length_smart': valid_max,
            'mean_top_20_length': mean_top_20,
            'length_used_for_error': representative_length, 
            'median_length': median_val,
            'max_spine_length': track_data['spine_length'].dropna().max(),       
            'median_spine_length': track_data['spine_length'].dropna().median(), 
            'mean_elevation_deg': track_data['elevation_deg'].mean(), 
            'gt': gt_val, 
            'abs_error_cm': abs_err,       
            'rel_error_%': rel_err,       
            'dist_camera_mean': track_data['fish_dist_from_camera'].mean()
        }
        resume_list.append(stats)
    return pd.DataFrame(resume_list)

PATH_MAPPINGS = {
    "/home/slimbook/bagfiles": "/home/rosuser/dataset/bagfiles",
    "/home/slimbook/fish_sizing/out": "/home/rosuser/repo/out",
    "/home/slimbook/models": "/home/rosuser/dataset/models/",
    "/home/slimbook/fish_sizing/config" :"/home/rosuser/repo/config/",
    "/media/slimbook/easystore": "/home/rosuser/easystore"
}

USE_DOCKER = True

def transform_path2docker(path: str) -> str:
    if not USE_DOCKER or path is None:
        return path
    for host_path, docker_path in PATH_MAPPINGS.items():
        if host_path in path:
            new_path = path.replace(host_path, docker_path)
            cprint(f"🔄 Path mapped: {path} \n   -> {new_path}", "yellow")
            return new_path 
    return path

def main():
    parser = argparse.ArgumentParser(description="Corrige is_3d_complete usando imágenes PNG y regenera CSVs.")
    parser.add_argument("--root_dir", "-root", default="/media/slimbook/easystore/results_fish_sizing/seleccio_article/lanty1/2025_08_21/1_peix/", help="Carpeta raíz a escanear")
    args = parser.parse_args()

    cprint(f"🔍 Escaneando árbol de directorios eficientemente en: {args.root_dir}", "cyan")
    
    args.root_dir = transform_path2docker(args.root_dir)
    raw_csvs = []
    for current_root, dirs, files in os.walk(args.root_dir):
        # 1. TRUCO DE OPTIMIZACIÓN EXTREMA: 
        # Le decimos a os.walk que NO entre a buscar dentro de las carpetas de frames ni en _inferred.
        # Esto reduce el tiempo de búsqueda de minutos a milisegundos en discos externos.
        dirs[:] = [d for d in dirs if not d.startswith("frame_") and d != "_inferred"]
        
        # 2. Si vemos una subcarpeta llamada 'results', comprobamos directamente si tiene el CSV
        if "results" in dirs:
            posible_csv = Path(current_root) / "results" / "all_fish_info_raw.csv"
            if posible_csv.exists():
                raw_csvs.append(posible_csv)
    
    if not raw_csvs:
        cprint("⚠️ No se encontraron archivos all_fish_info_raw.csv", "yellow")
        return

    cprint(f"📋 Encontrados {len(raw_csvs)} archivos RAW para corregir.\n", "green")

    for csv_path in raw_csvs:
        cprint(f"🚀 Procesando: {csv_path.parent}", "magenta")
        results_dir = csv_path.parent
        base_dir = results_dir.parent  
        
        df = pd.read_csv(csv_path)
        if df.empty: continue
            
        unique_frames = df['frame_id'].dropna().unique()
        
        for f_id in unique_frames:
            pkl_path = os.path.join(base_dir, f"{f_id}.pkl")
            # TU BRILLANTE IDEA: Usar el PNG guardado
            disp_png_path = os.path.join(base_dir, f"{f_id}_disparity.png")
            
            if not os.path.exists(pkl_path) or not os.path.exists(disp_png_path): 
                continue
                
            with open(pkl_path, 'rb') as f:
                frame_scene = pickle.load(f)
            
            # --- MAGIA CON EL PNG ---
            # 1. Cargamos la imagen en escala de grises (blanco y negro)
            disp_img = cv2.imread(disp_png_path, cv2.IMREAD_GRAYSCALE)
            
            # 2. Todo lo que NO sea negro estricto (> 5 por si hay compresión), lo convertimos en "Dato válido" (1000)
            # Esto engaña perfectamente a is_complete para que sepa dónde hay 3D
            disp_map_fake = np.where(disp_img > 5, 1000.0, 0.0).astype(np.float32)
            
            for fish in frame_scene.fish_list:
                if fish.mask is not None:
                    # Comprobamos la completitud usando nuestro mapa falso
                    fish.is_complete(disp_map_fake, debug_mode=False)
                    
                    # Actualizamos Pandas
                    mask = (df['frame_id'] == f_id) & (df['track_id'] == fish.track_id)
                    df.loc[mask, 'is_3D_complete'] = fish.is_3d_complete

        # Guardar RAW corregido
        df.to_csv(csv_path, index=False)
        cprint(f"   ✅ all_fish_info_raw.csv parcheado usando PNGs.", "green")

        # --- REGENERAR CSVS ---
        gt_val = df['gt'].dropna().iloc[0] if 'gt' in df.columns and not df['gt'].dropna().empty else None
        
        # Forzar tipos para evitar fallos tontos de Pandas
        for col in ['is_3D_complete', 'in_image_borders', 'does_overlap', 'is_front_fish']:
            if col in df.columns: df[col] = df[col].astype(bool)
            
        for col in ['filtered_length', 'aspect_ratio', 'elevation_deg']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

        # Filtro estricto
        filtered_result_df = df[
            (df['is_3D_complete'] == True) &
            (df['in_image_borders'] == False) &
            (df['does_overlap'] == False) &
            (df['is_front_fish'] == True) &
            (df['aspect_ratio'] >= 2) & 
            (df['filtered_length'].notna()) &
            (df['filtered_length'] > 0)
        ].copy()
         
        filtered_result_df.to_csv(os.path.join(results_dir, 'all_complete_ok_fish_2.csv'), index=False)

        if not filtered_result_df.empty:
            # Resumen raw
            resume_raw_df = filtered_result_df.groupby('track_id').agg(
                length_mean=('raw_length', 'mean'),
                length_max=('raw_length', 'max'),
                entry_count=('track_id', 'count')
            )
            resume_raw_df.to_csv(os.path.join(results_dir, 'resume_raw.csv'))

            # Resumen smart
            resume_filtered_df = filter_outliers_per_track(filtered_result_df, gt_val)
            resume_filtered_df.to_csv(os.path.join(results_dir, 'resume_filtered_smart.csv'), index=False)
            
            # Filtro por ángulo
            df_angle_ok = filtered_result_df[
                filtered_result_df['elevation_deg'].notna() & 
                (filtered_result_df['elevation_deg'].abs() <= 20.0)
            ].copy()
            df_angle_ok.to_csv(os.path.join(results_dir, 'all_angle_ok_fish_2.csv'), index=False)

            if not df_angle_ok.empty:
                filter_outliers_per_track(df_angle_ok, gt_val).to_csv(os.path.join(results_dir, 'resume_filtered_smart_angle_ok.csv'), index=False)
                cprint(f"   ✅ CSVs de resultados re-generados con éxito.", "green")
            else:
                cprint(f"   ⚠️ Se generaron CSVs, pero ninguno superó el filtro de Ángulo <= 20º", "yellow")
        else:
            cprint("   ⚠️ Sigue sin haber peces que pasen el filtro estricto.", "red")

if __name__ == "__main__":
    main()