import pandas as pd
import numpy as np
from pathlib import Path

import results_analysis_utils as rutils 

# =====================================================================
# CALCULATE FINAL RESULTS FROM RAW
# =====================================================================
def load_and_filter_dataset(path, dataset_name, recalculate_failure_reason=False, aspect_ratio_thr=3.0, angle_thr=30.0, save=False):
    
    path = Path(path)
    
    if not path.exists():
        print(f"⚠️ File not found: {path}")
        return pd.DataFrame(), pd.DataFrame()
        
    df = pd.read_csv(path)
    df['dataset'] = dataset_name
    
    # Eliminate not tracked fish
    df = df[df['track_id'] != -1].copy()
    
    if recalculate_failure_reason==True:
        df = rutils.assign_failure_reasons(df, aspect_ratio_thr=aspect_ratio_thr, angle_thr=angle_thr)
    
    # ---------------------------------------------------------
    # Crear ID único por track a prueba de fallos
    # ---------------------------------------------------------
    video_day_str = df['video_day'].astype(str) if 'video_day' in df.columns else "unknown_day"
    video_name_str = df['video_name'].astype(str) if 'video_name' in df.columns else "unknown_video"
    df['track_uid'] = df['dataset'] + "_" + video_day_str + "_" + video_name_str + "_" + df['track_id'].astype(str)
    
    total_initial_tracks = df['track_uid'].nunique()
    
    print(f"\n--- Processing {dataset_name} ---")
    print(f"Total raw frames with valid GT: {len(df)}")
    print(f"Total initial tracks: {total_initial_tracks}")
    
    # 1. Filtrado por frames
    track_failures = df.groupby('track_uid').apply(rutils.track_failure_from_frames).rename('track_failure_reason')

    measured_uids = track_failures[track_failures == 'measured'].index
    
    df_measured = df[
        (df['track_uid'].isin(measured_uids)) & 
        (df['failure_reason'] == 'measured')].copy()
    
    tracks_not_measured = total_initial_tracks - len(measured_uids)
    print(f"Frames belonging to 'measured' tracks: {len(df_measured)}")
    print(f"📉 Tracks descartados por 'track_failure_from_frames': {tracks_not_measured}")
    
    if df_measured.empty:
        print(f"❌ ERROR: Ningún track superó el filtro 'measured'.")
        return pd.DataFrame(), df
    
    # 2. Aplicar smart agg pasándole los umbrales
    track_metrics = df_measured.groupby('track_uid').apply(
        rutils.smart_aggregator,
        aspect_ratio_thr=aspect_ratio_thr,
        angle_thr=angle_thr
    ).dropna(how='all')
                
    tracks_after_agg = len(track_metrics)
    tracks_discarded_by_agg = len(measured_uids) - tracks_after_agg
    print(f"📉 Tracks descartados por 'smart_aggregator': {tracks_discarded_by_agg}")
    
    if track_metrics.empty:
        print(f"❌ ERROR: Todos los tracks fueron descartados por el aggregator.")
        return pd.DataFrame(), df
    
    valid_final_uids = track_metrics.index.unique()
    short_track_uids = set(measured_uids) - set(valid_final_uids)
    
    mask_short = (df['track_uid'].isin(short_track_uids)) & (df['failure_reason'] == 'measured')
    df.loc[mask_short, 'failure_reason'] = 'short_track'
    
    # =================================================================
    # 3. Recuperar Metadata y Distancia Media
    # =================================================================
    cols_to_keep = ['dataset']
    for col in ['scenario', 'video_day', 'video_name', 'especie_gt', 'fish_id']:
        if col in df_measured.columns:
            cols_to_keep.append(col)
        
    track_metadata = df_measured.groupby('track_uid').first()[cols_to_keep]
    
    # Calcular la distancia media a la cámara para cada pez
    col_distancia = 'fish_dist_from_camera' # <-- CÁMBIALO por el nombre de tu columna si es distinto
    if col_distancia in df_measured.columns:
        distancias_medias = df_measured.groupby('track_uid')[col_distancia].mean().rename('mean_distance_to_camera')
        track_metadata = track_metadata.join(distancias_medias)
    
    final_tracks = track_metrics.join(track_metadata).reset_index()
    
    # Porcentaje de error relativo
    if 'abs_error_cm' in final_tracks.columns and 'gt_cm' in final_tracks.columns:
        final_tracks['rel_error_perc'] = (final_tracks['abs_error_cm'] / final_tracks['gt_cm']) * 100
    else:
        final_tracks['rel_error_perc'] = np.nan
        
    # =================================================================
    # 4. Final (Las líneas que faltaban y causaban el error NoneType)
    # =================================================================
    success_rate = (len(final_tracks) / total_initial_tracks) * 100
    print(f"✅ FINAL RESULT: {len(final_tracks)} independent FISH (tracks) successfully aggregated.")
    print(f"📊 Success Rate: {success_rate:.2f}% ({len(final_tracks)}/{total_initial_tracks} tracks medidos con éxito)")
    
    if save:
        output_csv_path = path.parent / f"{dataset_name}_MEASUREMENTS.csv"
        final_tracks.to_csv(output_csv_path, index=False)
        print(f"💾 CSV Guardado con éxito en: {output_csv_path}")
    
    # IMPORTANTE: Este es el return que le faltaba a la función
    return final_tracks, df


# =====================================================================
# EJECUCIÓN DEL SCRIPT
# =====================================================================
if __name__ == "__main__":
    # Ajusta estas rutas a tus archivos reales
    ASPECT_RATIO_THR = 3.0
    ANGLE_THR = 30.0

   
        
    csv_path = Path("/media/slimbook/easystore3/results_fish_sizing/seleccio_article/SARMIENTO/2025/07_46_28/2025-07-16-07-46-28_0/results/all_fish_info_raw.csv")
   
    
    
    
    df_resultado,df = load_and_filter_dataset(
        csv_path, 
        dataset_name = "SEA_video_1"
    )
    

    # Mostrar un avance de los resultados
    if not df_resultado.empty:
        print("\n📊 Vista previa de las métricas finales:")
        print(f"   MAE:  {df_resultado['abs_error_cm'].mean():.2f} cm")
        print(f"   MAPE: {df_resultado['rel_error_perc'].mean():.2f} %")