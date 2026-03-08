
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from termcolor import cprint

from pathlib import Path
import pandas as pd


def create_df_from_root_folders(root_dir, output_csv_path, results_dir="corrected_results", gt=None):

    root_dir = Path(root_dir)
    pattern = f"**/{results_dir}/*_raw.csv"

    print(f"Buscando datos RAW en: {root_dir}")

    all_dfs = []
    df_raw=[]

    for filepath in root_dir.glob(pattern):

        print("Procesando csv: ",filepath)

        try:
            df = pd.read_csv(filepath)

            if df.empty:
                cprint("WARNING!!! CSV EMPTY!!! ","yellow")

            parts = filepath.parts
            idx = parts.index(results_dir)
            video_folder = parts[idx-4]
            video_day = parts[idx-4]
            folder_code = "-".join(parts[idx-3:idx-1])
            print("FOLDER CODE IS: ",folder_code)
            # folder_code = filepath.stem.replace("_raw", "")
            df["video_day"] = video_day
            df["source_folder"] = folder_code
            all_dfs.append(df)

        except Exception as e:
            print(f"❌ Error en {filepath}: {e}")

    if not all_dfs:
        print("⚠️ No se encontró ningún archivo '_raw.csv'")
        return pd.DataFrame()

    # concatenar todo
    df_raw = pd.concat(all_dfs, ignore_index=True)

    print(f"✅ {len(df_raw)} detecciones cargadas desde {len(all_dfs)} archivos")

    if gt is not None:
        df_raw["gt"] = gt

    cols = ["video_day"] + ["source_folder"] + [c for c in df_raw.columns if c != "source_folder" and c != "video_day"]
    df_raw = df_raw[cols]

    output_path = Path(output_csv_path) / f"{video_folder}_raw_aggregated.csv"

    df_raw.to_csv(output_path, index=False)

    print(f"💾 DataFrame guardado en: {output_path}")

    return df_raw


def classify_failure(row, ASPECT_RATIO_THR=3):
    if row["filtered_length"] > 0:
        return "measured"
    
    elif row["in_image_borders"]:
        return "borders"
    
    elif row["does_overlap"]:
        return "overlap"
    
    elif not row["is_3D_complete"]:
            return "incomplete_3D"
    
    elif row["aspect_ratio"] < ASPECT_RATIO_THR:
        return "aspect_ratio_fail"
    
    else:
        return "other"
    

def plot_tracks_failure_distribution(df_raw_agg, aspect_ratio_thr=3.0, show_global=True, figsize_video=(12,6), figsize_global=(5,5)):
    """
    Genera gráficos de distribución de tracks medidos y causas de fallo.

    Retorna:
    --------
    track_failures : pd.DataFrame
        DataFrame con cada track y su 'track_failure_reason' y 'source_folder'.
    """
    
    failure_priority = ["measured", "incomplete_3D", "borders", "overlap", "aspect_ratio_fail", "other"]

    def track_failure_from_frames(track_df):
        if "measured" in track_df["failure_reason"].values:
            return "measured"
        for reason in failure_priority[1:]:
            if reason in track_df["failure_reason"].values:
                return reason
        return "other"

    # Tracks por prioridad de fallo
    track_failures = df_raw_agg.groupby("unique_track").apply(track_failure_from_frames).reset_index()
    track_failures.columns = ["unique_track", "track_failure_reason"]
    track_failures["source_folder"] = track_failures["unique_track"].apply(lambda x: "_".join(x.split("_")[:-1]))

    # Conteo apilado por video
    track_stacked_df = track_failures.groupby(["source_folder", "track_failure_reason"]).size().unstack(fill_value=0)
    for col in failure_priority:
        if col not in track_stacked_df.columns:
            track_stacked_df[col] = 0
    track_stacked_df = track_stacked_df[failure_priority]

    # Gráfico apilado por video
    track_stacked_df.plot(kind="bar", stacked=True, figsize=figsize_video,
                          color=["#4CAF50", "#FFB74D", "#FF8A65", "#E57373", "#BA68C8", "#90A4AE"])
    plt.xlabel("Video Code")
    plt.ylabel("Número de Tracks")
    plt.title("Distribución apilada de tracks medidos y causas de fallo por vídeo")
    plt.xticks(rotation=45)
    plt.legend(title="Causa de fallo")
    plt.tight_layout()
    plt.show()

    if show_global:
        global_counts = track_failures["track_failure_reason"].value_counts().reindex(failure_priority, fill_value=0)

        plt.figure(figsize=figsize_global)
        bottom = 0
        colors = ["#4CAF50", "#FFB74D", "#FF8A65", "#E57373", "#BA68C8", "#90A4AE"]

        for i, reason in enumerate(failure_priority):
            count = global_counts[reason]
            plt.bar("Global", count, bottom=bottom, color=colors[i], label=reason)
            # Añadimos número encima de cada segmento
            plt.text(0, bottom + count/2, str(count), ha="center", va="center", color="white", fontweight="bold")
            bottom += count

        plt.ylabel("Número de Tracks")
        plt.title("Distribución global de tracks medidos y causas de fallo")
        plt.legend(title="Causa de fallo", bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()

    return track_failures


def plot_frame_failures(df_raw_agg, figsize_video=(12,6), figsize_global=(5,5), show_global=True):
    """
    Plots stacked bar charts of frame-level failures per video and globally.

    Parameters
    ----------
    df_raw_agg : pd.DataFrame
        DataFrame containing at least 'source_folder' and 'failure_reason' columns.
    figsize_video : tuple
        Figure size for per-video stacked plot.
    figsize_global : tuple
        Figure size for global stacked plot.
    show_global : bool
        If True, shows the global stacked bar plot.
        
    Returns
    -------
    stacked_df : pd.DataFrame
        DataFrame with counts of frames per video per failure reason.
    """
    # 1️⃣ Priority of failure reasons
    failure_priority = ["measured", "incomplete_3D", "borders", "overlap", "aspect_ratio_fail", "other"]

    # 2️⃣ Aggregate frames per video code
    stacked_df = df_raw_agg.groupby(["source_folder", "failure_reason"]).size().unstack(fill_value=0)

    # Ensure all columns exist
    for col in failure_priority:
        if col not in stacked_df.columns:
            stacked_df[col] = 0

    stacked_df = stacked_df[failure_priority]

    # 3️⃣ Plot per-video stacked bars
    stacked_df.plot(kind="bar", stacked=True, figsize=figsize_video,
                    color=["#4CAF50", "#FFB74D", "#FF8A65", "#E57373", "#BA68C8", "#90A4AE"])
    plt.xlabel("Video Code")
    plt.ylabel("Número de Frames")
    plt.title("Distribución apilada de mediciones y causas de fallo por vídeo")
    plt.xticks(rotation=45)
    plt.legend(title="Causa de fallo")
    plt.tight_layout()
    plt.show()

    # 4️⃣ Global stacked bar
    if show_global:
        global_counts = df_raw_agg["failure_reason"].value_counts().reindex(failure_priority, fill_value=0)
        global_df = global_counts.reset_index()
        global_df.columns = ["failure_reason", "count"]

        print("Conteo global de frames por causa:")
        print(global_df)

        plt.figure(figsize=figsize_global)
        bottom = 0
        colors = ["#4CAF50", "#FFB74D", "#FF8A65", "#E57373", "#BA68C8", "#90A4AE"]

        for i, row in global_df.iterrows():
            plt.bar(
                x=["Global"], 
                height=[row["count"]],
                bottom=bottom,
                color=colors[i],
                label=row["failure_reason"] if bottom == 0 else ""
            )
            bottom += row["count"]

        plt.ylabel("Número de Frames")
        plt.title("Distribución global de frames medidos y causas de fallo")
        plt.legend(title="Causa de fallo", bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()

    return stacked_df


def plot_aspect_ratio_vs_length(df_raw, ASPECT_RATIO_THR=3.0, figsize=(10, 8)):
    """
    Genera un gráfico doble mostrando:
    1️⃣ Scatter de filtered_length vs aspect_ratio
    2️⃣ Histograma de distribución de aspect_ratio

    Parámetros
    ----------
    df_raw : pd.DataFrame
        DataFrame con columnas 'aspect_ratio', 'filtered_length', 'gt'.
    ASPECT_RATIO_THR : float
        Umbral mínimo de aspect ratio para marcar descartes.
    figsize : tuple
        Tamaño de la figura (ancho, alto).

    Retorna
    -------
    None
    """
    # Filtramos valores válidos para evitar ceros o nulos
    df_valid = df_raw[(df_raw['aspect_ratio'] > 0) & (df_raw['filtered_length'] > 0)].copy()

    if df_valid.empty:
        print("⚠️ No hay datos válidos para graficar.")
        return

    # Figura con 2 subplots (arriba scatter, abajo histograma)
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # ==========================================
    # PANEL SUPERIOR: Medida vs Aspect Ratio
    # ==========================================
    sns.scatterplot(data=df_valid, x='aspect_ratio', y='filtered_length', 
                    alpha=0.4, color='blue', edgecolor=None, ax=axes[0])

    # Línea del Ground Truth (GT)
    if 'gt' in df_valid.columns and not df_valid['gt'].isnull().all():
        gt_medio_m = df_valid['gt'].mean() / 100.0  # convertir cm a metros si aplica
        axes[0].axhline(y=gt_medio_m, color='green', linestyle='-', linewidth=2.5, 
                        label=f'Ground Truth ({gt_medio_m:.3f} m)')

    # Línea del umbral de Aspect Ratio
    axes[0].axvline(x=ASPECT_RATIO_THR, color='red', linestyle='--', linewidth=2,
                    label=f'Umbral AR ({ASPECT_RATIO_THR})')

    axes[0].set_title("Efecto del Aspect Ratio en la Medición 3D (Filtered Length)", fontsize=13)
    axes[0].set_ylabel("Longitud Medida (m)")
    axes[0].legend(loc='upper right')

    # ==========================================
    # PANEL INFERIOR: Histograma de Frecuencia
    # ==========================================
    sns.histplot(data=df_valid, x='aspect_ratio', bins=50, kde=True, color="teal", ax=axes[1])
    axes[1].axvline(x=ASPECT_RATIO_THR, color='red', linestyle='--', linewidth=2)
    axes[1].set_ylabel("Nº de Frames")
    axes[1].set_xlabel("Aspect Ratio (Ancho BBox / Alto BBox)")

    plt.tight_layout()
    plt.show()
    
    
def smart_aggregator(track_df,num_tracks_threshold=5,deviation_from_median=1.5):
    """Smart filtering to all tracks"""
    num_frames = len(track_df)
    
    # if There's not a min number of tracks do not consider this measurement
    if num_frames < num_tracks_threshold: return None
    
    sorted_lengths = track_df['filtered_length'].dropna().sort_values(ascending=False).tolist()
    if not sorted_lengths: return None
    
    valid_max = sorted_lengths[0]
    if len(sorted_lengths) >= 5:
        while len(sorted_lengths) > 2:
            c_max, n_max = sorted_lengths[0], sorted_lengths[1]
            c_med = np.median(sorted_lengths)
            
            # If maximum is further than c_med * deviation_from_median of the median consider it an outlier
            if c_max > c_med * deviation_from_median:
                sorted_lengths.pop(0); continue
            # if maximum is more thabn a 5% further than the next measure discard it
            if (c_max - n_max) / n_max < 0.05:
                valid_max = c_max; break
            else:
                sorted_lengths.pop(0)
        if len(sorted_lengths) <= 2: valid_max = sorted_lengths[0]
            
    if num_frames < 20: 
        rep_len = valid_max
    else:
        n_top = max(1, int(len(sorted_lengths) * 0.2))
        rep_len = sum(sorted_lengths[:n_top]) / n_top
        
    gt_val = track_df['gt'].iloc[0]
    rel_err = abs((rep_len - (gt_val/100.0)) * 100) / gt_val * 100 if gt_val > 0 else None
    
    return pd.Series({
        'n_frames_validos': num_frames,
        'length_used_for_error': rep_len,
        'mean_elevation_deg': track_df['elevation_deg'].mean(),
        'mean_aspect_ratio': track_df['aspect_ratio'].mean(),
        'gt': gt_val,
        'rel_error_%': rel_err
    })

def plot_ablation_pipeline(df_raw, ASPECT_RATIO_THR=3, ANGLE_THR=20):
    """
    Realiza un estudio de ablation sobre filtros de tracks:
    - Filtros paso a paso: 3D completo, sin bordes, sin solapamiento, aspect ratio, ángulo
    - Aplica smart_aggregator a cada etapa
    - Muestra resumen tabular y gráficos de impacto
    
    Parámetros
    ----------
    df_raw : pd.DataFrame
        Debe contener columnas:
        ['source_folder','track_id','filtered_length','gt',
         'elevation_deg','aspect_ratio','is_3D_complete',
         'in_image_borders','does_overlap']
    ASPECT_RATIO_THR : float
        Umbral mínimo de aspect ratio para aceptar un frame
    ANGLE_THR : float
        Umbral máximo de ángulo Z para aceptar un frame
    """
    import warnings
    import matplotlib.lines as mlines
    warnings.filterwarnings('ignore')

    # PASO 0: Frames válidos base
    df_base = df_raw[df_raw['filtered_length'] > 0].copy()

    # Definir máscaras paso a paso
    mask_1 = df_base['is_3D_complete'] == True
    mask_2 = mask_1 & (df_base['in_image_borders'] == False)
    mask_3 = mask_2 & (df_base['does_overlap'] == False)
    mask_4 = mask_3 & (df_base['aspect_ratio'] >= ASPECT_RATIO_THR)
    mask_5 = mask_4 & df_base['elevation_deg'].notna() & (df_base['elevation_deg'].abs() <= ANGLE_THR)

    # Aplicar smart_aggregator por track en cada etapa
    stages_dict = {
        "0. Base (HDBSCAN sin filtros)": df_base.groupby(['source_folder','track_id']).apply(rutils.smart_aggregator).dropna().reset_index(),
        "1. + Filtro: is_3D_complete": df_base[mask_1].groupby(['source_folder','track_id']).apply(rutils.smart_aggregator).dropna().reset_index(),
        "2. + Filtro: Sin Bordes": df_base[mask_2].groupby(['source_folder','track_id']).apply(rutils.smart_aggregator).dropna().reset_index(),
        "3. + Filtro: Sin Solapamiento": df_base[mask_3].groupby(['source_folder','track_id']).apply(rutils.smart_aggregator).dropna().reset_index(),
        f"4. + Filtro: Aspect Ratio >= {ASPECT_RATIO_THR}": df_base[mask_4].groupby(['source_folder','track_id']).apply(rutils.smart_aggregator).dropna().reset_index(),
        f"5. + Filtro: Ángulo Z <= {ANGLE_THR}º": df_base[mask_5].groupby(['source_folder','track_id']).apply(rutils.smart_aggregator).dropna().reset_index()
    }

    # Contar frames activos en cada paso
    frames_count = [len(df_base), mask_1.sum(), mask_2.sum(), mask_3.sum(), mask_4.sum(), mask_5.sum()]

    # Imprimir resumen tabular
    print(f"{'ETAPA DEL PIPELINE (CASCADA)':<45} | {'ERROR MEDIO':<15} | {'FRAMES VIVOS':<15} | {'PECES (TRACKS)'}")
    print("-"*100)
    for (name, df_stage), f_count in zip(stages_dict.items(), frames_count):
        if df_stage.empty:
            print(f"{name:<45} | --- VACÍO ---")
            continue
        mean_err = df_stage['abs_error_cm'].mean()
        std_err = df_stage['abs_error_cm'].std()
        total_tracks = len(df_stage)
        print(f"{name:<45} | {mean_err:>5.2f} ± {std_err:>4.2f} cm | {f_count:>7} frames | {total_tracks:>5} tracks")

    # ====================================================
    # Gráficos de impacto
    # ====================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    stage_names = ["0. Base", "+ 3D_ok", "+ No Bordes", "+ No Solape", "+ AspectRatio", "+ ÁnguloZ"]

    # --- Panel izquierdo: Error absoluto ---
    plot_data = []
    for df_stage, short_name in zip(stages_dict.values(), stage_names):
        if not df_stage.empty:
            temp_df = pd.DataFrame({'Etapa': short_name, 'Error Absoluto (cm)': df_stage['abs_error_cm']})
            plot_data.append(temp_df)

    if plot_data:
        df_plot = pd.concat(plot_data)
        sns.boxplot(data=df_plot, x='Error Absoluto (cm)', y='Etapa', palette="viridis", ax=axes[0],
                    showmeans=True, meanprops={"marker":"D", "markerfacecolor":"white",
                                               "markeredgecolor":"black","markersize":7})
    axes[0].set_title("Evolución del Error Absoluto al aplicar filtros", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Error Absoluto (cm)")
    axes[0].set_ylabel("")
    mean_legend = mlines.Line2D([], [], color='white', marker='D', markeredgecolor='black', markersize=7, label='Error Medio (Tabla)')
    axes[0].legend(handles=[mean_legend], loc='lower right')

    # --- Panel derecho: Retención de datos ---
    tracks_count = [len(df) for df in stages_dict.values()]
    ax1 = axes[1]
    ax2 = ax1.twinx()
    ax1.plot(stage_names, frames_count, color='#d62728', marker='o', linewidth=2.5, label='Frames Útiles')
    ax2.bar(stage_names, tracks_count, color='#1f77b4', alpha=0.4, label='Peces Retenidos (Tracks)')

    ax1.set_ylabel("Cantidad de Frames (Línea Roja)", color='#d62728', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#d62728', colors='#d62728') 

    ax2.set_ylabel("Cantidad de Peces (Barras Azules)", color='#1f77b4', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#1f77b4', colors='#1f77b4') 

    axes[1].set_title("Retención de Datos tras cada filtro", fontsize=14, fontweight='bold')
    ax1.set_xticklabels(stage_names, rotation=25, ha="right")

    # Combinar leyendas
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.tight_layout()
    plt.show()
    
    return stages_dict