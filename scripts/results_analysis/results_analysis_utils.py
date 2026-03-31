import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from termcolor import cprint
from matplotlib.ticker import ScalarFormatter, MultipleLocator, MaxNLocator
from tqdm import tqdm
import warnings
import matplotlib.lines as mlines

warnings.filterwarnings('ignore')

# =====================================================================
# FUNCIONES DE AGREGACIÓN
# =====================================================================

def load_and_filter_dataset(path, dataset_name,recalculate_failure_reason=False, aspect_ratio_thr=3.0, angle_thr=30.0, save=False):
    if not path.exists():
        print(f"⚠️ File not found: {path}")
        return pd.DataFrame(), pd.DataFrame()
        
    df = pd.read_csv(path)
    df['dataset'] = dataset_name
    
    # 0. Quitar los Ground Truth no válidos y los tracks huérfanos (-1)
    df = df[df['gt'] > 0].copy()
    df = df[df['track_id'] != -1].copy()
    
    if recalculate_failure_reason==True:
        df = assign_failure_reasons(df, aspect_ratio_thr=aspect_ratio_thr, angle_thr=angle_thr)
    
    # Crear ID único por track
    df['track_uid'] = df['dataset'] + "_" + df['video_day'].astype(str) + "_" + df['video_name'].astype(str) + "_" + df['track_id'].astype(str)
    
    total_initial_tracks = df['track_uid'].nunique()
    
    print(f"\n--- Processing {dataset_name} ---")
    print(f"Total raw frames with valid GT: {len(df)}")
    print(f"Total initial tracks: {total_initial_tracks}")
    
    # 1. Filtrado por frames
    track_failures = df.groupby('track_uid').apply(track_failure_from_frames).rename('track_failure_reason')

    measured_uids = track_failures[track_failures == 'measured'].index
    
    # Aquí ahora estamos 100% seguros de que 'measured' significa lo que nosotros queremos
    df_measured = df[
        (df['track_uid'].isin(measured_uids)) & 
        (df['failure_reason'] == 'measured')].copy()
    
    tracks_not_measured = total_initial_tracks - len(measured_uids)
    print(f"Frames belonging to 'measured' tracks: {len(df_measured)}")
    print(f"📉 Tracks descartados por 'track_failure_from_frames': {tracks_not_measured}")
    
    if df_measured.empty:
        print(f"❌ ERROR: Ningún track superó el filtro 'measured'.")
        return pd.DataFrame(), df
    
    # 2. Aplicar smart agg    
    track_metrics = df_measured.groupby('track_uid').apply(smart_aggregator).dropna(how='all')
                
    tracks_after_agg = len(track_metrics)
    tracks_discarded_by_agg = len(measured_uids) - tracks_after_agg
    print(f"📉 Tracks descartados por 'smart_aggregator': {tracks_discarded_by_agg}")
    
    if track_metrics.empty:
        print(f"❌ ERROR: Todos los tracks fueron descartados por el aggregator.")
        return pd.DataFrame(), df
    
    # Sacamos los IDs de los tracks que han sobrevivido al agregador
    valid_final_uids = track_metrics.index.unique()
    
    # Los que estaban en 'measured_uids' pero NO en 'valid_final_uids' son los cortos
    short_track_uids = set(measured_uids) - set(valid_final_uids)
    
    # En el dataframe crudo, cambiamos su estado de 'measured' a 'short_track'
    mask_short = (df['track_uid'].isin(short_track_uids)) & (df['failure_reason'] == 'measured')
    df.loc[mask_short, 'failure_reason'] = 'short_track'
    # =================================================================
    
    # 3. Recuperar Metadata
    cols_to_keep = ['dataset', 'scenario', 'video_day', 'video_name']
    if 'especie_gt' in df_measured.columns:
        cols_to_keep.append('especie_gt')
    elif 'fish_id' in df_measured.columns:
        cols_to_keep.append('fish_id')
        
    track_metadata = df_measured.groupby('track_uid').first()[cols_to_keep]
    final_tracks = track_metrics.join(track_metadata).reset_index()
    
    # Porcentaje de error relativo
    final_tracks['rel_error_perc'] = (final_tracks['abs_error_cm'] / final_tracks['gt_cm']) * 100
    
    success_rate = (len(final_tracks) / total_initial_tracks) * 100
    print(f"✅ FINAL RESULT: {len(final_tracks)} independent FISH (tracks) successfully aggregated.")
    print(f"📊 Success Rate: {success_rate:.2f}% ({len(final_tracks)}/{total_initial_tracks} tracks medidos con éxito)")
    
    if save:
        # GUARDAR EL CSV DE ESTE DATASET
        output_csv_path = path.parent / f"{dataset_name}_FINAL_aggregated_metrics.csv"
        final_tracks.to_csv(output_csv_path, index=False)
    
    # Devolvemos TANTO el final como el RAW (para poder contar los totales luego)
    return final_tracks, df


def aggregate_results_from_root_new(root_dir, day_code, output_csv_path=None, results_foldername="corrected_results"):
    """
    Recursively searches ONLY for '*raw.csv' files. 
    Smartly extracts the video name directly from the filename, making it immune to folder structure changes.
    """
    root_dir = Path(root_dir)
    pattern = f"**/{results_foldername}/*raw.csv"
    cprint(f"🔍 Buscando datos RAW en: {root_dir}", "cyan")

    result_files = list(root_dir.rglob(pattern))

    if not result_files:
        cprint(f"❌ No se encontraron archivos '*raw.csv'.", "red")
        return pd.DataFrame()

    cprint(f"📂 Encontrados {len(result_files)} archivos de resultados crudos.", "green")
    all_dfs = []

    for filepath in result_files:
        try:
            # filepath.parts contiene toda la ruta separada
            # Ej multi: ... / 13_14_42 / 13-14-43_0 / corrected_results / archivo.csv
            # Ej single: ... / 10_48_44 / 0 / results_article_basic / corrected_results / archivo.csv
            
            parent_folder = filepath.parts[-3] # La carpeta justo encima de 'corrected_results'
            
            # Si el parent folder es "results_article..." (pasa en single_fish)
            if "results" in parent_folder.lower():
                # Cogemos las dos carpetas por encima (Ej: 10_48_44 y 0)
                folder_name = filepath.parts[-5]
                sub_id = filepath.parts[-4]
                video_name = f"{folder_name}_{sub_id}"
            else:
                # Es multiple_fish, el parent_folder es exactamente la llave del GT (Ej: 13-14-43_0)
                video_name = parent_folder
            
            df = pd.read_csv(filepath)
            
            if not df.empty:
                df['video_day'] = day_code
                df['video_name'] = video_name
                df['source_folder'] = video_name 
                all_dfs.append(df)
                
        except Exception as e:
            cprint(f"   ❌ Error leyendo {filepath}: {e}", "red")

    if not all_dfs:
        return pd.DataFrame()

    agg_df = pd.concat(all_dfs, ignore_index=True)

    # Reordenar columnas
    context_cols = ["video_day", "source_folder", "video_name"]
    other_cols = [c for c in agg_df.columns if c not in context_cols]
    agg_df = agg_df[context_cols + other_cols]

    cprint(f"\n📊 TOTAL AGREGADO: {len(agg_df)} líneas de datos.", "magenta", attrs=["bold"])
    return agg_df


def aggregate_all_results(output_base_dir, day_code=""):
    """
    Busca recursivamente todos los csv de resultados y genera una agregación global.
    """
    cprint(f"🔍 Buscando resultados en: {output_base_dir}", "cyan")
    result_files = list(Path(output_base_dir).rglob("all_fish_info_raw.csv"))
    
    if not result_files:
        cprint("❌ No se encontraron archivos 'all_fish_info_raw.csv'.", "red")
        return

    cprint(f"📂 Encontrados {len(result_files)} archivos de resultados.", "green")
    all_dfs = []
    
    for csv_path in result_files:
        video_name = csv_path.parent.parent.name
        df = pd.read_csv(csv_path)
        
        if not df.empty:
            df['source_folder'] = video_name
            df["video_day"] = day_code
            all_dfs.append(df)
            cprint(f"   ✅ Cargado: {video_name} ({len(df)} detecciones)", "white")
        else:
            cprint(f"   ⚠️ Vacío: {video_name}", "yellow")

    if not all_dfs:
        cprint("❗ No hay datos válidos para agregar.", "yellow")
        return pd.DataFrame()

    agg_df = pd.concat(all_dfs, ignore_index=True)
    save_path = os.path.join(output_base_dir, "agg_all_fish_results.csv")
    agg_df.to_csv(save_path, index=False)
    
    cprint(f"\n📊 TOTAL AGREGADO: {len(agg_df)} líneas de datos.", "magenta", attrs=["bold"])
    cprint(f"💾 Guardado en: {save_path}", "green")
    
    return agg_df

def aggregate_results_from_root(root_dir, day_code, output_csv_path=None, results_foldername="results", csv_name="all_fish_info_raw.csv", gt=None):
    """
    Busca recursivamente todos los csv dentro de carpetas 'results' sin importar su nivel de profundidad.
    """
    root_dir = Path(root_dir)
    pattern = f"**/{results_foldername}/{csv_name}"
    cprint(f"🔍 Buscando datos RAW en: {root_dir}", "cyan")

    result_files = list(root_dir.rglob(pattern))

    if not result_files:
        cprint(f"❌ No se encontraron archivos '{csv_name}'.", "red")
        return pd.DataFrame()

    cprint(f"📂 Encontrados {len(result_files)} archivos de resultados.", "green")
    all_dfs = []

    for filepath in result_files:
        try:
            video_name = filepath.parent.parent.name
            rel_path = filepath.relative_to(root_dir)
            source_folder = str(rel_path.parent.parent).replace(os.sep, "_") 
            
            df = pd.read_csv(filepath)
            
            if not df.empty:
                df['video_day'] = day_code
                df['source_folder'] = source_folder 
                df['video_name'] = video_name       
                all_dfs.append(df)
            else:
                cprint(f"   ⚠️ Vacío: {video_name}", "yellow")
                
        except Exception as e:
            cprint(f"   ❌ Error leyendo {filepath}: {e}", "red")

    if not all_dfs:
        return pd.DataFrame()

    agg_df = pd.concat(all_dfs, ignore_index=True)

    if gt is not None:
        agg_df["gt"] = gt
        if "filtered_length" in agg_df.columns:
            agg_df["abs_error_cm"] = abs(agg_df["filtered_length"] * 100 - agg_df["gt"])

    context_cols = ["video_day", "source_folder", "video_name"]
    other_cols = [c for c in agg_df.columns if c not in context_cols]
    agg_df = agg_df[context_cols + other_cols]

    if output_csv_path is None: output_csv_path = root_dir
        
    output_path = Path(output_csv_path) / f"{day_code}_raw_aggregated.csv"
    os.makedirs(output_path.parent, exist_ok=True)
    agg_df.to_csv(output_path, index=False)

    cprint(f"\n📊 TOTAL AGREGADO: {len(agg_df)} líneas de datos.", "magenta", attrs=["bold"])
    return agg_df


######################################################################33
#               READ  GT
#####################################################################333

def inject_ground_truth(df_raw, gt_dict, measures_dict, drop_unlabeled=False):
    """
    Inyecta el Ground Truth en el DataFrame, recalcula el error absoluto para todos
    (incluidos los -100) y marca los errores como 'not_a_fish'.
    
    Parámetros:
    - drop_unlabeled: Si es True, elimina del DataFrame los peces que no 
      estén en tu diccionario de anotaciones.
    """
    df = df_raw.copy()
    
    # 0. Limpiamos cualquier rastro de GT anterior antes de empezar
    cols_to_drop = ["gt", "abs_error_cm", "especie_gt", "failure_reason"]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    # 1. Convertir los diccionarios a un DataFrame plano
    records = []
    for day, videos in gt_dict.items():
        for video_name, tracks in videos.items():
            for track_id, label in tracks.items():
                records.append({
                    "video_day": day,
                    "video_name": video_name,
                    "track_id": track_id,
                    "especie_gt": label,
                    "gt": measures_dict.get(label, np.nan) 
                })
                
    df_gt = pd.DataFrame(records)
            
    # 2. Cruzar los datos con tu DataFrame original
    df = df.merge(df_gt, on=["video_day", "video_name", "track_id"], how="left")
    
    # 3. Marcar los que son basura ("error" o -100) en failure_reason
    if "failure_reason" not in df.columns:
        df["failure_reason"] = np.nan
        
    # Buscamos la palabra "error" en la columna de la etiqueta, o el valor -100
    mask_not_fish = (df["especie_gt"] == "error") | (df["gt"] == -100)
    df.loc[mask_not_fish, "failure_reason"] = "not_a_fish"
    
    # 4. Calcular error absoluto 
    df["abs_error_cm"] = abs((df["filtered_length"] * 100) - df["gt"])
    
    # 5. Crear el ID único ANTES de filtrar para poder contar los tracks
    df['unique_track'] = df['video_day'].astype(str) + "/" + df['video_name'].astype(str) + "/" + df['track_id'].astype(str)
    
    # 6. Filtrar los NO etiquetados (Opcional)
    if drop_unlabeled:
        df = df[df["gt"].notna()].copy()
        cprint(f"   🗑️ Se han eliminado los tracks sin etiquetar (drop_unlabeled=True).", "magenta")
    
    # 1. Sacamos los nombres exactos de los tracks huérfanos usando .unique()
    lista_tracks_sin_gt = df[df['gt'].isna()]['unique_track'].unique().tolist()
    
    # 2. Sacamos la cantidad (la longitud de esa lista)
    num_tracks_sin_gt = len(lista_tracks_sin_gt)
    
    # 3. Contamos los que SÍ tienen GT
    num_tracks_con_gt = df[df['gt'].notna()]['unique_track'].nunique()
    
    video_codes = df["source_folder"].unique()
    
    cprint(f"✅ Ground Truth inyectado. Vídeos únicos: {len(video_codes)}", "green")
    cprint(f"   🐟 Tracks CON Ground Truth: {num_tracks_con_gt}", "cyan")
    
    if num_tracks_sin_gt > 0:
        cprint(f"   ⚠️ Tracks SIN Ground Truth (NaN): {num_tracks_sin_gt}", "yellow")
        # Ahora sí, si haces un print aquí, verás la lista de nombres
        # print(lista_tracks_sin_gt) 
    else:
        cprint(f"   🎉 ¡Todos los tracks tienen su Ground Truth!", "green", attrs=["bold"])

    # 6. Filtrar los NO etiquetados (Opcional)
    if drop_unlabeled:
        df = df[df["gt"].notna()].copy()
        cprint(f"   🗑️ Se han eliminado los tracks sin etiquetar (drop_unlabeled=True).", "magenta")
    
    # Devolvemos el DataFrame y la LISTA REAL de tracks sin etiquetar
    return df, lista_tracks_sin_gt

    


# =====================================================================
# ⚡ CLASIFICADOR VECTORIZADO DINÁMICO (Reemplaza a classify_failure)
# =====================================================================

def assign_failure_reasons(df, aspect_ratio_thr=3.0, angle_thr=20.0):
    """
    Calcula la causa de fallo de manera rápida (vectorizada).
    Evalúa los umbrales estrictos en vivo, sobrescribiendo el estado original.
    """
    df = df.copy()

    c_not_fish = df["gt"] == -100
    c_borders  = df["in_image_borders"]
    c_overlap  = df["does_overlap"]
    c_ar       = df["aspect_ratio"] < aspect_ratio_thr
    c_3d       = df["is_3D_complete"] == False
    
    # # Manejo seguro por si en algún csv antiguo no existe is_3D_complete
    # c_3d = (df["is_3D_complete"] == False) if "is_3D_complete" in df.columns else pd.Series(False, index=df.index)

    if "elevation_deg" in df.columns and angle_thr is not None:
        c_angle = df["elevation_deg"].abs() > angle_thr
    else:
        c_angle = pd.Series(False, index=df.index)

    c_bad_cloud = pd.Series(False, index=df.index)
    if "pointcloud_size_ok" in df.columns and "fish_3d_ok" in df.columns:
        # Usamos ~ (NOT) para detectar cuando son Falsos
        c_bad_cloud = (~df["pointcloud_size_ok"].astype(bool)) | (~df["fish_3d_ok"].astype(bool))

    # 3. Éxito
    c_measured = df["filtered_length"] > 0

    # 4. ORDEN DE PRIORIDAD EN LA CASCADA
    conds = [c_not_fish, c_borders, c_overlap, c_ar, c_angle, c_3d, c_bad_cloud, c_measured]

    choices = [
        "not_a_fish",
        "borders",
        "overlap",
        "aspect_ratio_fail",
        "angle_fail",
        "incomplete_3D",
        "bad_pointcloud",  
        "measured"
    ]

    df["failure_reason"] = np.select(conds, choices, default="other")

    return df


# =====================================================================
# GRÁFICOS Y ANÁLISIS
# =====================================================================

def track_failure_from_frames_0(track_df):
    failure_priority = ["measured", "incomplete_3D", "borders", "overlap", "aspect_ratio_fail", "angle_fail","short_track","other"]
    
    
    if "measured" in track_df["failure_reason"].values:
        return "measured"
    for reason in failure_priority[1:]:
        if reason in track_df["failure_reason"].values:
            return reason
    return "other"

def get_primary_track_failure(track_df):
    # 1️⃣ Filtro implacable: Si el tracker lo perdió antes de 6 frames, 
    # la causa raíz SIEMPRE es que el track es demasiado corto.
    if len(track_df) <= min_frames:
        return 'short_track'
    
    # 2️⃣ Si el track es suficientemente largo, miramos si consiguió el mínimo de medidas buenas
    measured_count = (track_df['failure_reason'] == 'measured').sum()
    if measured_count > min_frames:
        return 'measured'
    
    # 3️⃣ Si es un track largo pero no llegó al mínimo de medidas buenas,
    # buscamos en qué falló principalmente (quitando los frames buenos aislados)
    bad_frames = track_df[track_df['failure_reason'] != 'measured']
    
    # Filtramos 'other' para no enmascarar errores físicos (bordes, ángulos, etc.)
    fallos_reales = bad_frames[bad_frames['failure_reason'] != 'other']['failure_reason']
    
    if not fallos_reales.empty:
        return fallos_reales.mode()[0]
        
    # Si llegamos aquí, es que todos los fallos eran literalmente 'other'
    return 'other'


def plot_tracks_failure_distribution(df_raw_agg_input, aspect_ratio_thr=3.0, angle_thr=20.0, 
                                     show_global=True, figsize_video=(12,6), figsize_global=(5,5),
                                     context_label =""):
    df_raw_agg = df_raw_agg_input.copy()
    
    # Clasificación en vivo
    df_raw_agg = assign_failure_reasons(df_raw_agg, aspect_ratio_thr, angle_thr)
    
    failure_priority = ["measured", "incomplete_3D", "borders", "overlap", "aspect_ratio_fail", "angle_fail","short_track" ,"other"]
    plot_colors = ["#4CAF50", "#FFB74D", "#FF8A65", "#E57373", "#BA68C8", "#F06292","#3d8f95", "#90A4AE"]
    
    

    track_failures = df_raw_agg.groupby("unique_track").apply(track_failure_from_frames).reset_index()
    track_failures.columns = ["unique_track", "track_failure_reason"]
    track_failures["source_folder"] = track_failures["unique_track"].apply(lambda x: x.split("/")[1])
    # track_failures["source_folder"] = track_failures["unique_track"].apply(lambda x: "_".join(x.split("_")[:-1]))

    track_stacked_df = track_failures.groupby(["source_folder", "track_failure_reason"]).size().unstack(fill_value=0)
    for col in failure_priority:
        if col not in track_stacked_df.columns: track_stacked_df[col] = 0
    track_stacked_df = track_stacked_df[failure_priority]

    track_stacked_df.plot(kind="bar", stacked=True, figsize=figsize_video, color=plot_colors)
    plt.xlabel("Video Code")
    plt.ylabel("Número de Tracks")
    plt.title(f"{context_label} Distribución apilada de tracks (AR >= {aspect_ratio_thr} | Ang <= {angle_thr}º)")
    plt.xticks(rotation=45)
    plt.legend(title="Causa de fallo", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    if show_global:
        global_counts = track_failures["track_failure_reason"].value_counts().reindex(failure_priority, fill_value=0)
        plt.figure(figsize=figsize_global)
        bottom = 0

        for i, reason in enumerate(failure_priority):
            count = global_counts[reason]
            label_with_count = f"{reason} ({count})"
            plt.bar("Global", count, bottom=bottom, color=plot_colors[i], label=label_with_count)
            bottom += count

        plt.ylabel("Número de Tracks")
        plt.title("Distribución global de tracks medidos")
        plt.legend(title="Causa de fallo", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        print("\n" + "="*60)
        print("📊 TABLA 1: Distribución Global de Tracks")
        print("="*60)
        global_df = global_counts.reset_index()
        global_df.columns = ["Causa de Fallo", "Cantidad"]
        global_df["Porcentaje (%)"] = (global_df["Cantidad"] / global_df["Cantidad"].sum() * 100).round(2)
        display(global_df) 

        print("\n" + "="*60)
        print("📊 TABLA 2: Desglose por Vídeo")
        print("="*60)
        display(track_stacked_df)

    return track_failures


def plot_frame_failures(df_raw_agg_input, aspect_ratio_thr=3.0, angle_thr=20.0, 
                        figsize_video=(12,6), figsize_global=(5,5), show_global=True,
                        context_label =""):
    
    df_raw_agg = df_raw_agg_input.copy()
    
    # Clasificación en vivo
    df_raw_agg = assign_failure_reasons(df_raw_agg, aspect_ratio_thr, angle_thr)

    failure_priority = ["measured", "incomplete_3D", "borders", "overlap", "aspect_ratio_fail", "angle_fail","short_track" ,"other"]
    plot_colors = ["#4CAF50", "#FFB74D", "#FF8A65", "#E57373", "#BA68C8", "#F06292","#3d8f95", "#90A4AE"]

    stacked_df = df_raw_agg.groupby(["source_folder", "failure_reason"]).size().unstack(fill_value=0)
    for col in failure_priority:
        if col not in stacked_df.columns: stacked_df[col] = 0
    stacked_df = stacked_df[failure_priority]

    stacked_df.plot(kind="bar", stacked=True, figsize=figsize_video, color=plot_colors)
    plt.xlabel("Video Code")
    plt.ylabel("Número de Frames")
    plt.title(f"Distribución apilada de frames (AR >= {aspect_ratio_thr} | Ang <= {angle_thr}º)")
    plt.xticks(rotation=45)
    plt.legend(title="Causa de fallo", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    if show_global:
        global_counts = df_raw_agg["failure_reason"].value_counts().reindex(failure_priority, fill_value=0)
        global_df = global_counts.reset_index()
        global_df.columns = ["Causa de Fallo", "Cantidad"]

        plt.figure(figsize=figsize_global)
        bottom = 0

        for i, row in global_df.iterrows():
            count = row["Cantidad"]
            label_with_count = f"{row['Causa de Fallo']} ({count})"
            plt.bar(["Global"], [count], bottom=bottom, color=plot_colors[i], label=label_with_count)
            bottom += count

        plt.ylabel("Número de Frames")
        plt.title(f"{context_label} Distribución global de frames medidos")
        plt.legend(title="Causa de fallo", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        print("\n" + "="*60)
        print("📊 TABLA 1: Distribución Global de Frames")
        print("="*60)
        global_df["Porcentaje (%)"] = (global_df["Cantidad"] / global_df["Cantidad"].sum() * 100).round(2)
        display(global_df)

    return stacked_df



def plot_frame_failures_0(df_raw_agg_input, aspect_ratio_thr=3.0, angle_thr=20.0, 
                        figsize_video=(12,6), figsize_global=(5,5), show_global=True,
                        context_label =""):
    
    df_raw_agg = df_raw_agg_input.copy()
    
    # Clasificación en vivo
    df_raw_agg = assign_failure_reasons(df_raw_agg, aspect_ratio_thr, angle_thr)
    df_raw_agg.loc[df_raw_agg["track_id"] == -1, "failure_reason"] = "not_tracked"

    failure_priority = ["measured","not_tracked", "incomplete_3D", "borders", "overlap", "aspect_ratio_fail", "angle_fail", "other"]
    plot_colors = ["#4CAF50","#DB1B1B", "#FFB74D", "#FF8A65", "#E57373", "#BA68C8", "#F06292", "#90A4AE"]
    
    not_tracked_ratio = (df_raw_agg["track_id"] == -1).mean() * 100
    print(f"🚫 Frames no traqueados: {not_tracked_ratio:.2f}%")

    stacked_df = df_raw_agg.groupby(["source_folder", "failure_reason"]).size().unstack(fill_value=0)
    for col in failure_priority:
        if col not in stacked_df.columns: stacked_df[col] = 0
    stacked_df = stacked_df[failure_priority]

    stacked_df.plot(kind="bar", stacked=True, figsize=figsize_video, color=plot_colors)
    plt.xlabel("Video Code")
    plt.ylabel("Número de Frames")
    plt.title(f"Distribución apilada de frames (AR >= {aspect_ratio_thr} | Ang <= {angle_thr}º)")
    plt.xticks(rotation=45)
    plt.legend(title="Causa de fallo", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    if show_global:
        global_counts = df_raw_agg["failure_reason"].value_counts().reindex(failure_priority, fill_value=0)
        global_df = global_counts.reset_index()
        global_df.columns = ["Causa de Fallo", "Cantidad"]

        plt.figure(figsize=figsize_global)
        bottom = 0

        for i, row in global_df.iterrows():
            count = row["Cantidad"]
            label_with_count = f"{row['Causa de Fallo']} ({count})"
            plt.bar(["Global"], [count], bottom=bottom, color=plot_colors[i], label=label_with_count)
            bottom += count

        plt.ylabel("Número de Frames")
        plt.title(f"{context_label} Distribución global de frames medidos")
        plt.legend(title="Causa de fallo", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        print("\n" + "="*60)
        print("📊 TABLA 1: Distribución Global de Frames")
        print("="*60)
        global_df["Porcentaje (%)"] = (global_df["Cantidad"] / global_df["Cantidad"].sum() * 100).round(2)
        display(global_df)

    return stacked_df

def plot_aspect_ratio_vs_length(df_raw, ASPECT_RATIO_THR=3.0, figsize=(10, 8),context_label=""):
    df_valid = df_raw[(df_raw['aspect_ratio'] > 0) & (df_raw['filtered_length'] > 0)].copy()

    if df_valid.empty:
        print("⚠️ No hay datos válidos para graficar.")
        return

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    sns.scatterplot(data=df_valid, x='aspect_ratio', y='filtered_length', alpha=0.4, color='blue', ax=axes[0])
    
    if 'gt' in df_valid.columns and not df_valid['gt'].isnull().all():
        gt_m = df_valid['gt'].mean() / 100.0 
        axes[0].axhline(y=gt_m, color='green', linestyle='-', linewidth=2.5, label=f'GT ({gt_m:.3f} m)')

    axes[0].axvline(x=ASPECT_RATIO_THR, color='red', linestyle='--', linewidth=2, label=f'Umbral AR ({ASPECT_RATIO_THR})')
    axes[0].set_title(f"{context_label} Efecto del Aspect Ratio en la Medición 3D (Filtered Length) ", fontsize=13)
    axes[0].set_ylabel("Longitud Medida (m)")
    axes[0].legend(loc='upper right')

    sns.histplot(data=df_valid, x='aspect_ratio', bins=50, kde=True, color="teal", ax=axes[1])
    axes[1].axvline(x=ASPECT_RATIO_THR, color='red', linestyle='--', linewidth=2)
    axes[1].set_ylabel("Nº de Frames")
    axes[1].set_xlabel("Aspect Ratio (Ancho BBox / Alto BBox)")

    plt.tight_layout()
    plt.show()
    
def smart_aggregator(track_df, length_col='filtered_length',num_tracks_threshold=5,deviation_from_median=1.5):
    num_frames = len(track_df)
    if num_frames <= num_tracks_threshold: return None
    
    if track_df.empty:
        return pd.DataFrame(columns=['source_folder', 'track_id', 'n_frames_validos', 'calculated_length_cm', 'gt_cm', 'abs_error_cm', 'mean_elevation_deg'])
    
    sorted_lengths = track_df[length_col].dropna().sort_values(ascending=False).tolist()
    if not sorted_lengths: return None
    
    valid_max = sorted_lengths[0]
    if len(sorted_lengths) >= 5:
        while len(sorted_lengths) > 2:
            c_max, n_max = sorted_lengths[0], sorted_lengths[1]
            c_med = np.median(sorted_lengths)
            
            if c_max > c_med * deviation_from_median:
                sorted_lengths.pop(0); continue
                
            if (c_max - n_max) / n_max < 0.05:
                valid_max = c_max; break
            else:
                sorted_lengths.pop(0)
        if len(sorted_lengths) <= 2: valid_max = sorted_lengths[0]
            
    if num_frames < 20: 
        rep_len_m = valid_max
    else:
        n_top = max(1, int(len(sorted_lengths) * 0.2))
        rep_len_m = sum(sorted_lengths[:n_top]) / n_top
        
    gt_cm = track_df['gt'].iloc[0]
    calculated_length_cm = rep_len_m * 100.0
    abs_err_cm = abs(calculated_length_cm - gt_cm) if gt_cm > 0 else None
    
    return pd.Series({
        'n_frames_validos': num_frames,
        'calculated_length_cm': calculated_length_cm,
        'gt_cm': gt_cm,
        'abs_error_cm': abs_err_cm,
        'mean_elevation_deg': track_df['elevation_deg'].mean()
    })
    

def plot_ablation_pipeline(df_raw, ASPECT_RATIO_THR=3, ANGLE_THR=20,context_label=""):
    
    df_base = df_raw[df_raw['filtered_length'] > 0].copy()

    # Quality filters
    # 1. Overlaps with image borders or other fish
    mask_1 = df_base['in_image_borders'] == False
    mask_2 = mask_1 & (df_base['does_overlap'] == False)
    
    # 2. Geometric reconstruction failures (Valid 3D)
    mask_3 = mask_2 & (df_base['is_3D_complete'] == True)
    
    # 3. Parametric quality / pose filters
    mask_4 = mask_3 & (df_base['aspect_ratio'] >= ASPECT_RATIO_THR)
    mask_5 = mask_4 & df_base['elevation_deg'].notna() & (df_base['elevation_deg'].abs() <= ANGLE_THR)

    # ---------------------------------------------------------
    # UPDATED DICTIONARY AND COUNTS
    # ---------------------------------------------------------
    stages_dict = {
        "0. Base (HDBSCAN without filters)": df_base,
        "1. + Filter: No Borders": df_base[mask_1],
        "2. + Filter: No Overlap": df_base[mask_2],
        "3. + Filter: is_3D_complete": df_base[mask_3],
        f"4. + Filter: Aspect Ratio >= {ASPECT_RATIO_THR}": df_base[mask_4],
        f"5. + Filter: Z-Angle <= {ANGLE_THR}º": df_base[mask_5]
    }

    # Count the number of True values (retained frames) at each step
    frames_count = [len(df_base), mask_1.sum(), mask_2.sum(), mask_3.sum(), mask_4.sum(), mask_5.sum()]
    agg_stages_dict = {}

    print(f"{'ETAPA DEL PIPELINE (CASCADA)':<45} | {'ERROR MEDIO':<15} | {'FRAMES VIVOS':<15} | {'PECES (TRACKS)'}")
    print("-"*100)
    
    for (name, df_filtered), f_count in zip(stages_dict.items(), frames_count):
        if df_filtered.empty:
            print(f"{name:<45} | --- VACÍO ---")
            agg_stages_dict[name] = pd.DataFrame()
            continue
            
        df_stage = df_filtered.groupby(['video_day', 'source_folder', 'track_id']).apply(smart_aggregator).dropna().reset_index()
        
        if df_stage.empty:
            print(f"{name:<45} | --- DESCARTADOS ---")
            agg_stages_dict[name] = pd.DataFrame()
            continue
            
        agg_stages_dict[name] = df_stage
        mean_err = df_stage['abs_error_cm'].mean()
        std_err = df_stage['abs_error_cm'].std()
        
        print(f"{name:<45} | {mean_err:>5.2f} ± {std_err:>4.2f} cm | {f_count:>7} frames | {len(df_stage):>5} tracks")

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    stage_names = ["0. Base", "+ No Bordes", "+ No Solape", "+ AspectRatio", "+ ÁnguloZ", "+ 3D_ok"]

    plot_data = []
    for df_stage, short_name in zip(agg_stages_dict.values(), stage_names):
        if not df_stage.empty:
            plot_data.append(pd.DataFrame({'Etapa': short_name, 'Error Absoluto (cm)': df_stage['abs_error_cm']}))

    if plot_data:
        sns.boxplot(data=pd.concat(plot_data), x='Error Absoluto (cm)', y='Etapa', palette="viridis", ax=axes[0],
                    showmeans=True, meanprops={"marker":"D", "markerfacecolor":"white", "markeredgecolor":"black"})
        
    axes[0].set_xscale('symlog', linthresh=20)
    axes[0].xaxis.set_major_formatter(ScalarFormatter())
    axes[0].set_xticks([0, 0.5, 1, 1.5, 2, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10, 20])
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_title(f"{context_label} Evolución del Error Absoluto al aplicar filtros", fontsize=14, fontweight='bold')
    axes[0].legend(handles=[mlines.Line2D([], [], color='white', marker='D', markeredgecolor='black', label='Error Medio')], loc='lower right')

    tracks_count = [len(df) for df in agg_stages_dict.values()]
    
    ax1 = axes[1]
    ax2 = ax1.twinx()
    ax1.plot(stage_names, frames_count, color='#d62728', marker='o', linewidth=2.5, label='Frames Útiles')
    ax2.bar(stage_names, tracks_count, color='#1f77b4', alpha=0.4, label='Peces Retenidos (Tracks)')

    ax1.set_ylabel("Cantidad de Frames (Rojo)", color='#d62728', fontweight='bold')
    ax2.set_ylabel("Cantidad de Peces (Azul)", color='#1f77b4', fontweight='bold')
    axes[1].set_title("Retención de Datos tras cada filtro", fontsize=14, fontweight='bold')
    ax1.set_xticklabels(stage_names, rotation=25, ha="right")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.tight_layout()
    plt.show()
    
    return stages_dict, agg_stages_dict

def plot_smart_filter_explanation_pro(
    df, 
    folder_code, 
    track_id, 
    length_col='filtered_length', 
    ar_thr=1.8, 
    angle_thr=20.0
):
    folder_str, track_str = str(folder_code), str(track_id)
    mask = (df['source_folder'].astype(str) == folder_str) & (df['track_id'].astype(str) == track_str)
    track_df = df[mask].copy()
    
    if track_df.empty: 
        return
        
    track_df['frame_num'] = track_df['frame_id'].astype(str).str.extract(r'(\d+)').astype(float).astype(int)
    track_df[length_col] = track_df[length_col].replace(-1, np.nan)
    valid_df = track_df.dropna(subset=[length_col]).sort_values(by='frame_num').copy()
    
    if len(valid_df) < 3: 
        return

    sorted_df = valid_df.sort_values(by=length_col, ascending=False).copy()
    sorted_df['status'] = 'Valid (Ignored)' 
    
    lengths = sorted_df[length_col].tolist()
    indices = sorted_df.index.tolist() 
    outliers_double, outliers_jump = [], []
    
    if len(lengths) >= 5:
        while len(lengths) > 2:
            c_max, n_max, c_med = lengths[0], lengths[1], np.median(lengths)
            if c_max > c_med * 2.0:
                outliers_double.append(indices.pop(0)); lengths.pop(0); continue
            if (c_max - n_max) / n_max >= 0.05:
                outliers_jump.append(indices.pop(0)); lengths.pop(0)
            else: 
                break 
                
    sorted_df.loc[outliers_double, 'status'] = 'Rejected (Peak > 2x Median)'
    sorted_df.loc[outliers_jump, 'status'] = 'Rejected (Jump > 5%)'
    
    used_indices = [indices[0]] if len(valid_df) < 20 else indices[:max(1, int(len(lengths) * 0.2))]
    sorted_df.loc[used_indices, 'status'] = 'Selected (Top 20% Averaged)'
    
    final_length = sorted_df.loc[used_indices, length_col].mean()
    gt_cm = valid_df['gt'].iloc[0]
    gt_m = gt_cm / 100.0 if gt_cm > 0 else None

    valid_df = valid_df.merge(sorted_df[['status']], left_index=True, right_index=True)

    # Detect invalid angle zones
    valid_df['angle_out'] = (valid_df['elevation_deg'].abs() > angle_thr)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f"Smart Aggregator Breakdown - [Track: {track_str}]", fontsize=16, fontweight='bold', y=0.98)
    
    gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1], height_ratios=[2, 1, 1], hspace=0.25, wspace=0.3)
    
    color_dict = {
        'Rejected (Peak > 2x Median)': '#d62728',
        'Rejected (Jump > 5%)': '#ff7f0e',
        'Selected (Top 20% Averaged)': '#2ca02c',
        'Valid (Ignored)': '#7f7f7f'
    }

    markers_dict = {
        'Rejected (Peak > 2x Median)': 'X',
        'Rejected (Jump > 5%)': 'X',
        'Selected (Top 20% Averaged)': '*',
        'Valid (Ignored)': 'o'
    }

    # Smaller points for valid measures
    size_map = valid_df['status'].map(lambda s: 40 if s == 'Valid (Ignored)' else 120)

    # --- 1. Temporal view ---
    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(len(valid_df)-1):
        if valid_df['angle_out'].iloc[i]:
            ax1.axvspan(valid_df['frame_num'].iloc[i],
                        valid_df['frame_num'].iloc[i+1],
                        color='red', alpha=0.08)

    sns.scatterplot(
        data=valid_df,
        x='frame_num',
        y=length_col,
        hue='status',
        style='status',
        palette=color_dict,
        markers=markers_dict,
        s=size_map,
        ax=ax1,
        legend=False
    )
    ax1.axhline(y=final_length, color='#2ca02c', linewidth=2.5, label=f'Final Estimate ({final_length:.3f} m)')
    if gt_m:
        ax1.axhline(y=gt_m, color='blue', linestyle='--', linewidth=2, label=f'GT ({gt_m:.3f} m)')

    ax1.set_title("1. Temporal View", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Length (m)")
    ax1.legend(loc='upper center', fontsize=10, ncol=2)

    # --- Aspect ratio ---
    ax_ar = fig.add_subplot(gs[1, 0], sharex=ax1)
    for i in range(len(valid_df)-1):
        if valid_df['angle_out'].iloc[i]:
            ax_ar.axvspan(valid_df['frame_num'].iloc[i],
                          valid_df['frame_num'].iloc[i+1],
                          color='red', alpha=0.08)
    ax_ar.plot(valid_df['frame_num'], valid_df['aspect_ratio'], marker='^', color='purple')
    ax_ar.axhline(y=ar_thr, color='black', linestyle=':', label=f'AR Threshold ({ar_thr})')
    ax_ar.set_title("Aspect Ratio", fontsize=12, fontweight='bold')
    ax_ar.set_ylabel("Aspect Ratio")
    ax_ar.set_xlabel("Frame Number")
    ax_ar.legend(loc='upper right', fontsize=9)

    # --- Angle ---
    ax_ang = fig.add_subplot(gs[2, 0], sharex=ax1)
    for i in range(len(valid_df)-1):
        if valid_df['angle_out'].iloc[i]:
            ax_ang.axvspan(valid_df['frame_num'].iloc[i],
                           valid_df['frame_num'].iloc[i+1],
                           color='red', alpha=0.08)
    ax_ang.plot(valid_df['frame_num'], valid_df['elevation_deg'], marker='v', color='brown')
    ax_ang.axhline(y=angle_thr, color='black', linestyle=':')
    ax_ang.axhline(y=-angle_thr, color='black', linestyle=':')
    ax_ang.set_title("Elevation Angle (deg)", fontsize=12, fontweight='bold')
    ax_ang.set_ylabel("Angle (deg)")
    ax_ang.set_xlabel("Frame Number")
    ax_ang.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_ang.legend([f'Angle Threshold ±{angle_thr}°'], loc='upper right', fontsize=9)

    # --- 2. Algorithmic view (right side) ---
    ax2 = fig.add_subplot(gs[:, 1])
    sorted_df['rank'] = np.arange(1, len(sorted_df) + 1)
    sorted_df['angle_out'] = valid_df['angle_out']

    for i in range(len(sorted_df)-1):
        if sorted_df['angle_out'].iloc[i]:
            ax2.axvspan(sorted_df['rank'].iloc[i],
                        sorted_df['rank'].iloc[i+1],
                        color='red', alpha=0.08)

    size_map_sorted = sorted_df['status'].map(lambda s: 40 if s == 'Valid (Ignored)' else 120)
    sns.scatterplot(
        data=sorted_df,
        x='rank',
        y=length_col,
        hue='status',
        style='status',
        palette=color_dict,
        markers=markers_dict,
        s=size_map_sorted,
        ax=ax2
    )
    ax2.plot(sorted_df['rank'], sorted_df[length_col], color='gray', alpha=0.3)
    ax2.axhline(y=final_length, color='#2ca02c', linewidth=2.5)
    if gt_m:
        ax2.axhline(y=gt_m, color='blue', linestyle='--', linewidth=2)
    ax2.set_title("2. Algorithmic View", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Length (m)")
    ax2.set_xlabel("Rank")
    ax2.legend(loc='upper right', fontsize=10, title="Status")

    # --- Zoom around GT ±2cm ---
    if gt_m:
        
        # Ajustar el límite Y basándose en los datos reales para no perder los outliers
        y_max = valid_df[length_col].max()
        y_min = min(valid_df[length_col].min(), gt_m if gt_m else valid_df[length_col].min())
        
        # Le damos un 5% de margen por arriba y por abajo
        margin = (y_max - y_min) * 0.05
        
        ax1.set_ylim(y_min - margin, y_max + margin)
        ax2.set_ylim(y_min - margin, y_max + margin)
        
    plt.tight_layout()
    plt.show()
    
def plot_thresholds_interaction(df_base, ar_range=None, angles_to_test=None, optimal_ar=3,context=""):
    
    # Clean up false positives so they don't affect the graphs
    if 'failure_reason' in df_base.columns:
        df_base = df_base[df_base['failure_reason'] != 'not_a_fish'].copy()
        
    if ar_range is None: ar_range = np.arange(1.0, 5, 0.2)
    if angles_to_test is None: angles_to_test = [15, 20, 25, 30,35,40,45, 60, 90]
        
    def evaluate_thresholds(df, ar_thr, angle_thr):
        df_filt = df[(df['aspect_ratio'] >= ar_thr) & (df['elevation_deg'].abs() <= angle_thr)]
        if df_filt.empty: return np.nan, 0, 0
        
        # Agrupamos y aplicamos el agregador
        df_agg = df_filt.groupby(['video_day', 'source_folder', 'track_id']).apply(smart_aggregator)
        
        if df_agg.empty: return np.nan, 0, 0

        if 'abs_error_cm' not in df_agg.columns:
            # Buscamos cómo se llaman tus columnas de GT y Longitud en este momento
            gt_col = 'gt' if 'gt' in df_agg.columns else ('gt_cm' if 'gt_cm' in df_agg.columns else None)
            len_col = 'calculated_length_cm' if 'calculated_length_cm' in df_agg.columns else ('filtered_length' if 'filtered_length' in df_agg.columns else None)
            
            if gt_col and len_col:
                # Si usas 'filtered_length' (metros), multiplicamos por 100. Si ya son cm, directo.
                if len_col == 'filtered_length':
                    df_agg['abs_error_cm'] = abs((df_agg[len_col] * 100) - df_agg[gt_col])
                else:
                    df_agg['abs_error_cm'] = abs(df_agg[len_col] - df_agg[gt_col])
            else:
                # Si no encuentra las columnas, devuelve NaN para no romper el bucle
                return np.nan, len(df_filt), len(df_agg)

        
        df_agg = df_agg.dropna(subset=['abs_error_cm'])
        
        return df_agg['abs_error_cm'].mean() if not df_agg.empty else np.nan, len(df_filt), len(df_agg)
        
        
    results = []
    for ang in tqdm(angles_to_test, desc="Evaluating Angles"):
        for ar in ar_range:
            err, frames, tracks = evaluate_thresholds(df_base, ar, ang)
            results.append({
                'Angle_Thr': "No Z-Filter" if ang==90 else f"Angle <= {ang}º", 
                'AR_Thr': ar, 
                'Error_cm': err, 
                'Frames': frames, 
                'Tracks': tracks
            })

    df_interaction = pd.DataFrame(results)
    
    # 3 horizontal plots (increase figure width to 22)
    
    ctx_str = f" ({context})" if context else ""
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    palette = sns.color_palette("Set1", n_colors=len(angles_to_test))

    # --- PANEL 1: Error Evolution ---
    sns.lineplot(data=df_interaction, x='AR_Thr', y='Error_cm', hue='Angle_Thr', marker='o', palette=palette, ax=axes[0])
    axes[0].axvline(x=optimal_ar, color='black', linestyle=':', label=f'Chosen AR ({optimal_ar})')
    axes[0].set_title(f"1. Interaction: Angle Effect on Error {ctx_str}", fontweight='bold')
    axes[0].set_ylabel("Mean Absolute Error (cm)")
    
    # --- PANEL 2: Frame Retention ---
    sns.lineplot(data=df_interaction, x='AR_Thr', y='Frames', hue='Angle_Thr', marker='s', palette=palette, ax=axes[1])
    axes[1].axvline(x=optimal_ar, color='black', linestyle=':')
    axes[1].set_title(f"2. Data Cost: Retained Frames{ctx_str}", fontweight='bold')
    axes[1].set_ylabel("Number of Frames")

    # --- PANEL 3: Track Retention (Real Fish) ---
    sns.lineplot(data=df_interaction, x='AR_Thr', y='Tracks', hue='Angle_Thr', marker='D', palette=palette, ax=axes[2])
    axes[2].axvline(x=optimal_ar, color='black', linestyle=':')
    axes[2].set_title(f"3. True Cost: Retained Tracks{ctx_str}", fontweight='bold')
    axes[2].set_ylabel("Number of Tracks")

    plt.tight_layout()
    plt.show()
    
    return df_interaction
    
def plot_track_evolution_detailed(df, folder_code, track_id, ar_thr=3, angle_thr=20.0, dev_median=1.5, zoom_margin_cm=5.0):
    folder_str, track_str = str(folder_code), str(track_id)
    mask = (df['source_folder'].astype(str) == folder_str) & (df['track_id'].astype(str) == track_str)
    fish_data = df[mask].copy()
    
    if fish_data.empty: 
        print(f"⚠️ No data found for Folder: {folder_str}, Track: {track_str}")
        return
        
    # Extraer numero de frame para el eje X
    fish_data['frame_num'] = fish_data['frame_id'].astype(str).str.extract(r'(\d+)').astype(float).astype(int)
    fish_data = fish_data.sort_values(by='frame_num')
    
    gt_cm = fish_data['gt'].iloc[0]
    gt_m = gt_cm / 100.0 if gt_cm > 0 else None
    
    # Máscara de frames que superan los filtros estrictos
    perfect_mask = (fish_data['is_3D_complete'].fillna(False).astype(bool)) & \
                   (fish_data['aspect_ratio'].fillna(0) >= ar_thr) & \
                   (fish_data['elevation_deg'].fillna(90).abs() <= angle_thr)
    
    bad_frames = fish_data[~perfect_mask]['frame_num']
    valid_df = fish_data[perfect_mask & (fish_data['filtered_length'] > 0)].copy()
    
    outliers_idx, used_indices, ignored_indices, final_calculated_length = [], [], [], np.nan
    
    # Simular la lógica de tu smart_aggregator
    if not valid_df.empty:
        lengths, idxs = zip(*sorted(zip(valid_df['filtered_length'], valid_df.index), key=lambda x: x[0], reverse=True))
        lengths, idxs = list(lengths), list(idxs)
        
        if len(lengths) >= 5:
            while len(lengths) > 2:
                if lengths[0] > np.median(lengths) * dev_median:
                    outliers_idx.append(idxs.pop(0)); lengths.pop(0); continue
                if (lengths[0] - lengths[1]) / lengths[1] > 0.05:
                    outliers_idx.append(idxs.pop(0)); lengths.pop(0)
                else: break
                    
        used_indices = [idxs[0]] if len(valid_df) < 20 else idxs[:max(1, int(len(lengths) * 0.2))]
        ignored_indices = [i for i in idxs if i not in used_indices]
        if used_indices: 
            final_calculated_length = valid_df.loc[used_indices, 'filtered_length'].mean()

    measurement_fail_frames = fish_data[fish_data['filtered_length'] == -1]['frame_num']
    fish_data[['raw_length', 'filtered_length']] = fish_data[['raw_length', 'filtered_length']].replace(-1, np.nan)

    # ==========================================
    # CREAR LA FIGURA Y LOS PANELES
    # ==========================================
    fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True, gridspec_kw={'height_ratios': [1.5, 2, 1, 1]})
    fig.suptitle(f"Track Evolution Analysis | Folder: {folder_str} | Track ID: {track_str}", fontsize=18, fontweight='bold', y=0.98)
    
    # --- PANELES 0 y 1: Longitud (Global y Zoom) ---
    for i, ax in enumerate(axes[:2]):
        ax.plot(fish_data['frame_num'], fish_data['raw_length'], marker='.', color='orange', alpha=0.4, label='Raw Length (No smoothing)')
        ax.plot(fish_data['frame_num'], fish_data['filtered_length'], color='gray', alpha=0.6, label='Filtered Length (Smoothed)')
        
        if ignored_indices: 
            ax.scatter(valid_df.loc[ignored_indices, 'frame_num'], valid_df.loc[ignored_indices, 'filtered_length'], color='#1f77b4', marker='s', zorder=3, label='Valid Frame (Ignored by aggregator)')
        if outliers_idx: 
            ax.scatter(valid_df.loc[outliers_idx, 'frame_num'], valid_df.loc[outliers_idx, 'filtered_length'], color='#ff7f0e', marker='X', s=100, zorder=4, label='Aggregator Outlier (Discarded peak)')
        if used_indices: 
            ax.scatter(valid_df.loc[used_indices, 'frame_num'], valid_df.loc[used_indices, 'filtered_length'], color='#2ca02c', marker='*', s=200, edgecolor='black', zorder=5, label='Averaged Frame (Used for final sizing)')
            ax.axhline(y=final_calculated_length, color='#2ca02c', linewidth=2.5, label=f'Final Calculated Length ({final_calculated_length*100:.1f} cm)')
        
        if not measurement_fail_frames.empty: 
            ax.scatter(measurement_fail_frames, [fish_data['filtered_length'].min()]*len(measurement_fail_frames), color='red', marker='X', s=100, label='Failed Measure (-1)')
        if gt_m: 
            ax.axhline(y=gt_m, color='green', linestyle='--', linewidth=2.5, label=f'Ground Truth ({gt_cm:.1f} cm)')

        ax.set_ylabel("Fish Length (m)", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    axes[0].set_title("1. Global View (All length detections)", fontsize=14, fontweight='bold')
    axes[1].set_title(f"2. Zoomed View (±{zoom_margin_cm} cm margin around measurement)", fontsize=14, fontweight='bold')
    
    # Aplicar zoom al panel 1
    zoom_m = zoom_margin_cm / 100.0
    if gt_m:
        axes[1].set_ylim(gt_m - zoom_m, gt_m + zoom_m)
    elif not np.isnan(final_calculated_length):
        axes[1].set_ylim(final_calculated_length - zoom_m, final_calculated_length + zoom_m)

    # --- PANEL 2: Aspect Ratio ---
    axes[2].set_title("3. Bounding Box Aspect Ratio", fontsize=14, fontweight='bold')
    axes[2].plot(fish_data['frame_num'], fish_data['aspect_ratio'], marker='^', color='purple', label='Frame Aspect Ratio')
    axes[2].axhline(y=ar_thr, color='black', linestyle=':', linewidth=2, label=f'Min Threshold ({ar_thr})')
    axes[2].set_ylabel("Aspect Ratio", fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # --- PANEL 3: Z-Angle (Elevation) ---
    axes[3].set_title("4. Fish Posture (Z-Angle)", fontsize=14, fontweight='bold')
    axes[3].plot(fish_data['frame_num'], fish_data['elevation_deg'], marker='v', color='brown', label='Frame Elevation Angle')
    axes[3].axhline(y=angle_thr, color='black', linestyle=':', linewidth=2, label=f'Max Threshold (±{angle_thr}º)')
    axes[3].axhline(y=-angle_thr, color='black', linestyle=':', linewidth=2)
    axes[3].set_ylabel("Angle (º)", fontsize=12, fontweight='bold')
    axes[3].set_xlabel("Video Frame Number", fontsize=14, fontweight='bold') # ETIQUETA EJE X AQUI ABAJO
    axes[3].grid(True, alpha=0.3)
    
    # --- DIBUJAR LAS BANDAS ROJAS DE DESCARTE Y AÑADIRLAS A LAS LEYENDAS ---
    # Dibujamos las bandas en todos los gráficos
    for ax in axes:
        for bf in bad_frames: 
            ax.axvspan(bf - 0.5, bf + 0.5, color='red', alpha=0.10)

    # Truco para añadir la banda roja a la leyenda (creamos un 'Patch' manual)
    red_patch = Patch(color='red', alpha=0.10, label='Discarded Frame (Failed Filters)')
    
    # Añadimos las leyendas (ubicadas fuera del gráfico a la derecha para no tapar datos)
    handles, labels = axes[1].get_legend_handles_labels()
    handles.append(red_patch)
    labels.append('Discarded Frame (Failed Filters)')
    axes[1].legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, title="Length Legend", title_fontsize='11')
    
    axes[2].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
    axes[3].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)

    plt.tight_layout()
    plt.show()
    
    
def plot_final_error_analysis(df_filtered, top_n_outliers=15, figsize=(18, 14), title="Análisis Profundo del Error en la Etapa Final (3D Complete)"):
    """
    Genera un panel de 4 gráficas para analizar en profundidad los errores
    de las mediciones finales retenidas.
    """
    if df_filtered.empty:
        print("⚠️ El DataFrame está vacío. No hay datos para graficar.")
        return

    # Trabajar con una copia para no alterar el original ni generar warnings
    df = df_filtered.copy()

    # 1. Reconstruir el 'unique_track' por si el agregador lo eliminó al agrupar
    if 'unique_track' not in df.columns:
        # Asumimos que existen estas columnas; usamos .get() o manejamos excepciones si faltan
        df['unique_track'] = df['video_day'].astype(str) + "/" + df['video_name'].astype(str) + "/" + df['track_id'].astype(str)

    # Configuración visual para gráficos con calidad de publicación
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)

    # ==========================================================
    # 📊 GRÁFICA 1: Boxplot de Error por Vídeo
    # ==========================================================
    col_video = 'source_folder' if 'source_folder' in df.columns else 'video_name'
    sns.boxplot(data=df, x='abs_error_cm', y=col_video, ax=axes[0, 0], palette='viridis')
    sns.stripplot(data=df, x='abs_error_cm', y=col_video, size=5, color=".3", linewidth=0, ax=axes[0, 0], alpha=0.6)
    axes[0, 0].set_title('1. Consistencia por Secuencia de Vídeo', fontweight='bold')
    axes[0, 0].set_xlabel('Error Absoluto (cm)')
    axes[0, 0].set_ylabel('Secuencia de Vídeo')

    # ==========================================================
    # 📊 GRÁFICA 2: Error Absoluto por Especie
    # ==========================================================
    if 'especie_gt' in df.columns:
        sns.boxplot(data=df, x='especie_gt', y='abs_error_cm', ax=axes[0, 1], palette='Set2')
        sns.swarmplot(data=df, x='especie_gt', y='abs_error_cm', color=".25", ax=axes[0, 1])
        axes[0, 1].set_title('2. Error según la Especie', fontweight='bold')
        axes[0, 1].set_xlabel('Especie (Ground Truth)')
    else:
        axes[0, 1].text(0.5, 0.5, 'Columna especie_gt no encontrada', ha='center', va='center')
        axes[0, 1].set_title('2. Error según la Especie (Sin Datos)')
        
    axes[0, 1].set_ylabel('Error Absoluto (cm)')

    # ==========================================================
    # 📊 GRÁFICA 3: Sesgo de Escala (Error vs Tamaño Real)
    # ==========================================================
    hue_col = 'especie_gt' if 'especie_gt' in df.columns else None
    sns.scatterplot(
        data=df, x='gt_cm', y='abs_error_cm', hue=hue_col, 
        size='n_frames_validos', sizes=(50, 400), alpha=0.7, ax=axes[1, 0], palette='Set1'
    )
    sns.regplot(data=df, x='gt_cm', y='abs_error_cm', scatter=False, ax=axes[1, 0], color='gray', line_kws={"linestyle":"--"})
    axes[1, 0].set_title('3. Sesgo de Escala: Error vs. Tamaño Real', fontweight='bold')
    axes[1, 0].set_xlabel('Tamaño Real del Pez (cm)')
    axes[1, 0].set_ylabel('Error Absoluto (cm)')
    if hue_col:
        axes[1, 0].legend(title="Especie & Frames", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    # ==========================================================
    # 📊 GRÁFICA 4: Los "Villanos" (Top Outliers)
    # ==========================================================
    top_outliers = df.nlargest(top_n_outliers, 'abs_error_cm')
    sns.barplot(data=top_outliers, x='abs_error_cm', y='unique_track', ax=axes[1, 1], palette='Reds_r')
    axes[1, 1].set_title(f'4. Top {top_n_outliers} Outliers (Tracks con Mayor Error)', fontweight='bold')
    axes[1, 1].set_xlabel('Error Absoluto (cm)')
    axes[1, 1].set_ylabel('ID Único del Track')

    plt.tight_layout()
    plt.show()


def plot_error_vs_geometry(df, ar_thr=3.0, angle_thr=30.0, context_label="Global", max_error_cm=None):
    """
    Plotea la relación entre el Error Absoluto y las variables geométricas (Z-Angle y Aspect Ratio).
    Muestra una vista cruda (con ruido) y una vista purgada (aislando variables).
    """
    # 1. Preparar datos base (quitamos NaNs de las columnas que necesitamos)
    df_plot = df.dropna(subset=['abs_error_cm', 'aspect_ratio', 'elevation_deg']).copy()
    if df_plot.empty:
        print(f"⚠️ No data to plot geometry errors for {context_label}")
        return

    # Usamos el ángulo en valor absoluto (desviación del centro)
    df_plot['abs_elevation_deg'] = df_plot['elevation_deg'].abs()

    # Si no se define un máximo para el eje Y, calculamos uno seguro (percentil 95 + margen)
    # para que los outliers extremos no aplasten la gráfica visualmente.
    if max_error_cm is None:
        max_error_cm = df_plot['abs_error_cm'].quantile(0.95) * 1.5

    # 2. Preparar los DataFrames filtrados (aislando variables)
    # Filtro estructural: quitamos ruido que no tiene que ver con la geometría fina
    structural_noise = ['not_a_fish', 'borders', 'overlap', 'incomplete_3D']
    valid_structure_mask = ~df_plot['failure_reason'].isin(structural_noise)

    # DataFrame puro para Ángulo: Estructura OK + Aspect Ratio OK
    df_clean_angle = df_plot[valid_structure_mask & (df_plot['aspect_ratio'] >= ar_thr)]
    
    # DataFrame puro para Aspect Ratio: Estructura OK + Ángulo OK
    df_clean_ar = df_plot[valid_structure_mask & (df_plot['abs_elevation_deg'] <= angle_thr)]

    # 3. Dibujar la Figura 2x2
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Error vs Geometry Analysis | {context_label}", fontsize=18, fontweight='bold', y=0.98)

    # -------------------------------------------------------------------
    # FILA 1: DATOS CRUDOS (RAW) - Con todo el ruido
    # -------------------------------------------------------------------
    # 1.1 Ángulo (Raw)
    sns.scatterplot(data=df_plot, x='abs_elevation_deg', y='abs_error_cm', alpha=0.1, ax=axes[0,0], color='gray')
    sns.regplot(data=df_plot, x='abs_elevation_deg', y='abs_error_cm', scatter=False, ax=axes[0,0], color='black', line_kws={'linestyle':'--'})
    axes[0,0].axvline(angle_thr, color='red', linestyle='--', linewidth=2, label=f'Threshold ({angle_thr}º)')
    axes[0,0].set_title("1A. Raw Data: Error vs Z-Angle", fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel("Absolute Error (cm)")
    axes[0,0].set_ylim(-0.5, max_error_cm)
    axes[0,0].legend()

    # 1.2 Aspect Ratio (Raw)
    sns.scatterplot(data=df_plot, x='aspect_ratio', y='abs_error_cm', alpha=0.1, ax=axes[0,1], color='gray')
    sns.regplot(data=df_plot, x='aspect_ratio', y='abs_error_cm', scatter=False, ax=axes[0,1], color='black', line_kws={'linestyle':'--'})
    axes[0,1].axvline(ar_thr, color='red', linestyle='--', linewidth=2, label=f'Threshold ({ar_thr})')
    axes[0,1].set_title("1B. Raw Data: Error vs Aspect Ratio", fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel("Absolute Error (cm)")
    axes[0,1].set_ylim(-0.5, max_error_cm)
    axes[0,1].set_xlim(0, df_plot['aspect_ratio'].quantile(0.99))
    axes[0,1].legend()

    # -------------------------------------------------------------------
    # FILA 2: DATOS FILTRADOS (ISOLATED) - El efecto real de la variable
    # -------------------------------------------------------------------
    # 2.1 Ángulo (Clean)
    sns.scatterplot(data=df_clean_angle, x='abs_elevation_deg', y='abs_error_cm', alpha=0.2, ax=axes[1,0], color='#3498DB')
    sns.regplot(data=df_clean_angle, x='abs_elevation_deg', y='abs_error_cm', scatter=False, ax=axes[1,0], color='black', line_kws={'linestyle':'--'})
    axes[1,0].axvline(angle_thr, color='red', linestyle='--', linewidth=2)
    axes[1,0].set_title("2A. Isolated Variable: Error vs Z-Angle\n(Only frames with valid AR and no structural noise)", fontsize=13, fontweight='bold')
    axes[1,0].set_xlabel("Absolute Z-Angle (º)")
    axes[1,0].set_ylabel("Absolute Error (cm)")
    axes[1,0].set_ylim(-0.5, max_error_cm)

    # 2.2 Aspect Ratio (Clean)
    sns.scatterplot(data=df_clean_ar, x='aspect_ratio', y='abs_error_cm', alpha=0.2, ax=axes[1,1], color='#9B59B6')
    sns.regplot(data=df_clean_ar, x='aspect_ratio', y='abs_error_cm', scatter=False, ax=axes[1,1], color='black', line_kws={'linestyle':'--'})
    axes[1,1].axvline(ar_thr, color='red', linestyle='--', linewidth=2)
    axes[1,1].set_title("2B. Isolated Variable: Error vs Aspect Ratio\n(Only frames with valid Angle and no structural noise)", fontsize=13, fontweight='bold')
    axes[1,1].set_xlabel("Aspect Ratio")
    axes[1,1].set_ylabel("Absolute Error (cm)")
    axes[1,1].set_ylim(-0.5, max_error_cm)
    axes[1,1].set_xlim(0, df_plot['aspect_ratio'].quantile(0.99))

    plt.tight_layout()
    plt.show()
    

def plot_error_vs_track_length(df_raw, df_final, context_label="Global", min_frames_thr=5, target_error_cm=1.0):
    """
    Plotea el error del track en función de cuántos frames válidos tiene.
    Utiliza el error REAL calculado por el smart_aggregator para los tracks válidos.
    """
    # 1. Contar cuántos frames válidos tiene cada track desde el RAW
    valid_frames = df_raw[df_raw['failure_reason'].isin(['measured', 'short_track'])].copy()

    if valid_frames.empty:
        print(f"⚠️ No valid frames to plot track length analysis for {context_label}")
        return

    # Sacamos solo el conteo (n_frames)
    track_counts = valid_frames.groupby('track_uid').size().reset_index(name='n_frames')

    # 2. TRACKS ACEPTADOS: Usamos el error real e inteligente de df_final
    if df_final is not None and not df_final.empty:
        accepted_errors = df_final[['track_uid', 'abs_error_cm']].copy()
        accepted_errors.rename(columns={'abs_error_cm': 'final_error'}, inplace=True)
        # Cruzamos para tener n_frames y el error inteligente
        accepted_stats = pd.merge(
            track_counts[track_counts['n_frames'] > min_frames_thr],
            accepted_errors, 
            on='track_uid', 
            how='inner'
        )
    else:
        accepted_stats = pd.DataFrame(columns=['track_uid', 'n_frames', 'final_error'])

    # 3. TRACKS RECHAZADOS (Short Tracks): Usamos la media bruta de sus pocos frames
    # Solo para demostrar visualmente por qué se descartaron
    rejected_uids = track_counts[track_counts['n_frames'] <= min_frames_thr]['track_uid']
    rejected_frames = valid_frames[valid_frames['track_uid'].isin(rejected_uids)]
    
    if not rejected_frames.empty:
        rejected_errors = rejected_frames.groupby('track_uid')['abs_error_cm'].mean().reset_index()
        rejected_errors.rename(columns={'abs_error_cm': 'final_error'}, inplace=True)
        rejected_stats = pd.merge(
            track_counts[track_counts['n_frames'] <= min_frames_thr],
            rejected_errors, 
            on='track_uid', 
            how='inner'
        )
    else:
        rejected_stats = pd.DataFrame(columns=['track_uid', 'n_frames', 'final_error'])

    # Juntamos ambos para poder trazar la línea de tendencia general
    all_stats = pd.concat([accepted_stats, rejected_stats], ignore_index=True)

    if all_stats.empty:
        return

    # 4. Dibujar la Gráfica
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # Plot de los rechazados (Rojo/Gris)
    if not rejected_stats.empty:
        plt.scatter(
            rejected_stats['n_frames'], rejected_stats['final_error'], 
            color='#e74c3c', alpha=0.6, s=60, edgecolor='black', linewidth=0.5,
            label=f'Discarded by Aggregator (≤ {min_frames_thr} frames)'
        )
    
    # Plot de los aceptados (Verde)
    if not accepted_stats.empty:
        plt.scatter(
            accepted_stats['n_frames'], accepted_stats['final_error'], 
            color='#2ecc71', alpha=0.7, s=80, edgecolor='black', linewidth=0.8,
            label=f'Accepted Tracks (Smart Aggregation)'
        )

    # Añadir línea de tendencia
    sns.regplot(
        data=all_stats, x='n_frames', y='final_error', 
        scatter=False, color='black', logx=True, truncate=False,
        line_kws={'linestyle':'--', 'linewidth': 2}, label='Error Trend'
    )

    # Líneas de referencia
    plt.axvline(x=min_frames_thr + 0.5, color='gray', linestyle=':', linewidth=2.5, label=f'Aggregator Threshold')
    plt.axhline(y=target_error_cm, color='blue', linestyle='-.', linewidth=2, alpha=0.6, label=f'Target Error ({target_error_cm} cm)')

    # Estética y Etiquetas
    plt.title(f"Impact of Tracking Length on Final Measurement Accuracy | {context_label}", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Number of Valid Frames in Track", fontsize=14, fontweight='bold')
    plt.ylabel("Track Final Absolute Error (cm)", fontsize=14, fontweight='bold')
    
    # Ajustar el eje Y y X
    max_y = all_stats['final_error'].quantile(0.95) * 1.5
    if max_y < target_error_cm * 2: max_y = target_error_cm * 3
    plt.ylim(-0.5, max_y)
    plt.xlim(0, all_stats['n_frames'].max() + 2)

    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()
    
    


def plot_track_length_histogram(df_raw, min_frames_thr=5, context_label="Global"):
    """
    Dibuja un histograma apilado mostrando la distribución de la longitud de los tracks
    (Total de frames detectados) y qué proporción de ellos fue Aceptada vs Rechazada.
    """
    if df_raw.empty:
        return
        
    # 1. Calcular estadísticas por Track
    # - total_frames: Cuántas veces apareció el pez en el tracker (longitud bruta)
    # - valid_frames: Cuántos de esos frames superaron los filtros geométricos
    track_stats = df_raw.groupby('track_uid').agg(
        total_frames=('frame_id', 'count'),
        valid_frames=('failure_reason', lambda x: x.isin(['measured', 'short_track']).sum())
    ).reset_index()

    # 2. Etiquetar si el track sobrevivió o no al smart_aggregator
    # Sobrevive si tiene estrictamente más frames válidos que el umbral
    label_accepted = f'Accepted (> {min_frames_thr} valid frames)'
    label_rejected = f'Rejected (≤ {min_frames_thr} valid frames)'
    
    track_stats['Status'] = np.where(
        track_stats['valid_frames'] > min_frames_thr,
        label_accepted,
        label_rejected
    )

    # 3. Dibujar el Histograma Apilado
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # Definir colores (Verde para aceptados, Rojo apagado para rechazados)
    palette = {
        label_accepted: '#2ecc71',
        label_rejected: '#e74c3c'
    }

    # Dibujamos el histograma. discrete=True fuerza a que cada número entero (1,2,3...) tenga su barra
    sns.histplot(
        data=track_stats,
        x='total_frames',
        hue='Status',
        multiple='stack',
        palette=palette,
        discrete=True,
        edgecolor='black',
        alpha=0.85
    )

    # Estética y Textos
    plt.title(f"Track Survival Rate based on Total Tracking Length | {context_label}", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Total Frames in Track (Tracker detections)", fontsize=13, fontweight='bold')
    plt.ylabel("Number of Tracks", fontsize=13, fontweight='bold')

    # Ajustar el eje X para que no se estropee si hay UN track de 300 frames que aplasta el gráfico
    # Cortamos el eje X en el percentil 98%
    max_x_val = int(track_stats['total_frames'].quantile(0.98))
    # Nos aseguramos de dar al menos 20 de margen
    plt.xlim(0, max(20, max_x_val)) 

    # Mejorar la leyenda
    sns.move_legend(plt.gca(), "upper right", title='Aggregator Decision', frameon=True)

    plt.tight_layout()
    plt.show()