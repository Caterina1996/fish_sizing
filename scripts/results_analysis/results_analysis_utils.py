
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from termcolor import cprint

from pathlib import Path
import pandas as pd
   
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
import warnings
warnings.filterwarnings('ignore')

def create_df_from_root_folders(root_dir, folder_code,output_csv_path, results_dir="corrected_results", gt=None):

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
            idx = parts.index(folder_code)
            video_seq = parts[idx+1]
            camera_id = parts[idx+2]

            print("FOLDER CODE IS: ",folder_code)
            # folder_code = filepath.stem.replace("_raw", "")
            df["source_folder"] = f"{video_seq}_{camera_id}"
            df["video_day"] = folder_code
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
        df_raw["abs_error_cm"] = abs(df_raw["filtered_length"]*100-df_raw["gt"])

    cols = ["video_day"] + ["source_folder"] + [c for c in df_raw.columns if c != "source_folder" and c != "video_day"]
    df_raw = df_raw[cols]

    output_path = Path(output_csv_path) / f"{folder_code}_raw_aggregated.csv"

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
    
    elif row["aspect_ratio"] < ASPECT_RATIO_THR:
        return "aspect_ratio_fail"
    
    elif not row["is_3D_complete"]:
        return "incomplete_3D"
    
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
    
    
def smart_aggregator(track_df, length_col='filtered_length',num_tracks_threshold=5,deviation_from_median=1.5):
    num_frames = len(track_df)
    # if There's not a min number of tracks do not consider this measurement
    if num_frames < num_tracks_threshold: return None
    
    if track_df.empty:
        cprint("PROBLEMAAAAAAAAAAAAAAAAAAAAAA!!!","red")
        # Si no han sobrevivido frames a este filtro, devolvemos un DataFrame vacío bien formateado
        return pd.DataFrame(columns=[
            'source_folder', 'track_id', 'n_frames_validos', 
            'calculated_length_cm', 'gt_cm', 'abs_error_cm', 'mean_elevation_deg'
        ])
    
    sorted_lengths = track_df[length_col].dropna().sort_values(ascending=False).tolist()
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



def plot_ablation_pipeline(df_raw, ASPECT_RATIO_THR=3, ANGLE_THR=20):
    import warnings
    import matplotlib.lines as mlines
    from matplotlib.ticker import ScalarFormatter
    warnings.filterwarnings('ignore')

    # PASO 0: Frames válidos base
    df_base = df_raw[df_raw['filtered_length'] > 0].copy()

    # Definir máscaras paso a paso (¡Corregido el paréntesis en mask_6!)
    mask_2 = df_base['in_image_borders'] == False
    mask_3 = mask_2 & (df_base['does_overlap'] == False)
    mask_4 = mask_3 & (df_base['aspect_ratio'] >= ASPECT_RATIO_THR)
    mask_5 = mask_4 & df_base['elevation_deg'].notna() & (df_base['elevation_deg'].abs() <= ANGLE_THR)
    mask_6 = mask_5 & (df_base['is_3D_complete'] == True)

    # Diccionario RAW (Frames)
    stages_dict = {
        "0. Base (HDBSCAN sin filtros)": df_base,
        "1. + Filtro: Sin Bordes": df_base[mask_2],
        "2. + Filtro: Sin Solapamiento": df_base[mask_3],
        f"3. + Filtro: Aspect Ratio >= {ASPECT_RATIO_THR}": df_base[mask_4],
        f"4. + Filtro: Ángulo Z <= {ANGLE_THR}º": df_base[mask_5],
        "5. + Filtro: is_3D_complete": df_base[mask_6]
    }

    # Contar frames activos en cada paso (¡Corregido el final!)
    frames_count = [len(df_base), mask_2.sum(), mask_3.sum(), mask_4.sum(), mask_5.sum(), mask_6.sum()]

    # NUEVO DICCIONARIO PARA GUARDAR LOS TRACKS AGRUPADOS
    agg_stages_dict = {}

    # Imprimir resumen tabular
    print(f"{'ETAPA DEL PIPELINE (CASCADA)':<45} | {'ERROR MEDIO':<15} | {'FRAMES VIVOS':<15} | {'PECES (TRACKS)'}")
    print("-"*100)
    
    for (name, df_filtered), f_count in zip(stages_dict.items(), frames_count):
        
        if df_filtered.empty:
            print(f"{name:<45} | --- VACÍO ---")
            agg_stages_dict[name] = pd.DataFrame()
            continue
            
        df_stage = df_filtered.groupby(['video_day', 'source_folder', 'track_id']).apply(smart_aggregator).dropna()
        
        if df_stage.empty:
            print(f"{name:<45} | --- DESCARTADOS POR EL AGGREGATOR ---")
            agg_stages_dict[name] = pd.DataFrame()
            continue
            
        df_stage = df_stage.reset_index()
        
        # GUARDAMOS EL DATAFRAME AGRUPADO PARA LAS GRÁFICAS
        agg_stages_dict[name] = df_stage
        
        mean_err = df_stage['abs_error_cm'].mean()
        std_err = df_stage['abs_error_cm'].std()
        total_tracks = len(df_stage)
        
        print(f"{name:<45} | {mean_err:>5.2f} ± {std_err:>4.2f} cm | {f_count:>7} frames | {total_tracks:>5} tracks")

    # ====================================================
    # Gráficos de impacto
    # ====================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    stage_names = ["0. Base", "+ No Bordes", "+ No Solape", "+ AspectRatio", "+ ÁnguloZ", "+ 3D_ok"]

    # --- Panel izquierdo: Error absoluto ---
    plot_data = []
    # ¡OJO! Ahora iteramos sobre agg_stages_dict (que tiene los tracks, no los frames)
    for df_stage, short_name in zip(agg_stages_dict.values(), stage_names):
        if not df_stage.empty:
            temp_df = pd.DataFrame({'Etapa': short_name, 'Error Absoluto (cm)': df_stage['abs_error_cm']})
            plot_data.append(temp_df)

    if plot_data:
        df_plot = pd.concat(plot_data)
        sns.boxplot(data=df_plot, x='Error Absoluto (cm)', y='Etapa', palette="viridis", ax=axes[0],
                    showmeans=True, meanprops={"marker":"D", "markerfacecolor":"white",
                                               "markeredgecolor":"black","markersize":7})
        
    axes[0].set_xscale('symlog', linthresh=20)
    axes[0].xaxis.set_major_formatter(ScalarFormatter())
    
    axes[0].set_xticks([0, 0.5, 1, 1.5, 2, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10, 50])
    axes[0].tick_params(axis='x', rotation=45)

    axes[0].set_title("Evolución del Error Absoluto al aplicar filtros", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Error Absoluto (cm)")
    axes[0].set_ylabel("")
    mean_legend = mlines.Line2D([], [], color='white', marker='D', markeredgecolor='black', markersize=7, label='Error Medio (Tabla)')
    axes[0].legend(handles=[mean_legend], loc='lower right')

    # --- Panel derecho: Retención de datos ---
    tracks_count = [len(df) for df in agg_stages_dict.values()]
    
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

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.tight_layout()
    plt.show()
    
    return stages_dict, agg_stages_dict
    
    

def plot_smart_filter_explanation_pro(df, folder_code, track_id, length_col='filtered_length', ar_thr=1.8, angle_thr=20.0):
    """
    Desgrana y visualiza el funcionamiento interno del 'Smart Aggregator' 
    con alineación temporal del Aspect Ratio y el Ángulo Z.
    """
    # 1. AISLAR DATOS Y LIMPIAR -1
    folder_str = str(folder_code)
    track_str = str(track_id)
    mask = (df['source_folder'].astype(str) == folder_str) & (df['track_id'].astype(str) == track_str)
    track_df = df[mask].copy()
    
    if track_df.empty:
        print(f"❌ ERROR: No se encontraron datos para la carpeta '{folder_str}' y track '{track_str}'")
        return
        
    track_df['frame_num'] = track_df['frame_id'].astype(str).str.extract(r'(\d+)').astype(float).astype(int)
    
    # --- LA MAGIA CONTRA EL -1 ---
    track_df[length_col] = track_df[length_col].replace(-1, np.nan)
    valid_df = track_df.dropna(subset=[length_col]).sort_values(by='frame_num').copy()
    
    num_frames = len(valid_df)
    if num_frames < 3:
        print("⚠️ El track tiene menos de 3 frames con medidas válidas. El Smart Aggregator lo descartaría.")
        return

    # 2. REPLICAR LA LÓGICA DEL ALGORITMO
    sorted_df = valid_df.sort_values(by=length_col, ascending=False).copy()
    sorted_df['status'] = 'Válido (Ignorado)' 
    
    lengths = sorted_df[length_col].tolist()
    indices = sorted_df.index.tolist() 
    
    outliers_doble = []
    outliers_salto = []
    
    if len(lengths) >= 5:
        while len(lengths) > 2:
            c_max, n_max = lengths[0], lengths[1]
            c_med = np.median(lengths)
            
            if c_max > c_med * 2.0:
                outliers_doble.append(indices.pop(0))
                lengths.pop(0)
                continue
                
            if (c_max - n_max) / n_max >= 0.05:
                outliers_salto.append(indices.pop(0))
                lengths.pop(0)
            else:
                break 
                
    sorted_df.loc[outliers_doble, 'status'] = 'Descartado (Pico Gigante > 2x Mediana)'
    sorted_df.loc[outliers_salto, 'status'] = 'Descartado (Salto > 5%)'
    
    used_indices = []
    if num_frames < 20:
        if indices: used_indices = [indices[0]]
    else:
        n_top = max(1, int(len(lengths) * 0.2))
        used_indices = indices[:n_top]
        
    sorted_df.loc[used_indices, 'status'] = 'Seleccionado (Top 20% Promediado)'
    
    final_length = sorted_df.loc[used_indices, length_col].mean()
    gt_cm = valid_df['gt'].iloc[0]
    gt_m = gt_cm / 100.0 if gt_cm > 0 else None

    valid_df = valid_df.merge(sorted_df[['status']], left_index=True, right_index=True)

    # 3. PLOTEAR: LAYOUT ASIMÉTRICO CON GRIDSPEC
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Disección del Filtro Smart Aggregator - [Track ID: {track_str}]", fontsize=16, fontweight='bold', y=0.98)
    
    # Creamos el Grid: 3 filas, 2 columnas. Columna izq más ancha.
    gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1], height_ratios=[2, 1, 1], hspace=0.1)
    
    color_dict = {
        'Descartado (Pico Gigante > 2x Mediana)': '#d62728', 
        'Descartado (Salto > 5%)': '#ff7f0e',                
        'Seleccionado (Top 20% Promediado)': '#2ca02c',      
        'Válido (Ignorado)': '#7f7f7f'                       
    }
    markers_dict = {
        'Descartado (Pico Gigante > 2x Mediana)': 'X',
        'Descartado (Salto > 5%)': 'X',
        'Seleccionado (Top 20% Promediado)': '*',
        'Válido (Ignorado)': 'o'
    }

    # ==========================================
    # COLUMNA IZQUIERDA: VISTAS TEMPORALES
    # ==========================================
    
    # PANEL A: Longitud vs Tiempo
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(data=valid_df, x='frame_num', y=length_col, hue='status', style='status', 
                    palette=color_dict, markers=markers_dict, s=150, ax=ax1, legend=False)
    
    ax1.axhline(y=final_length, color='#2ca02c', linestyle='-', linewidth=2.5, label=f'Medida Final ({final_length:.3f} m)')
    if gt_m is not None:
        ax1.axhline(y=gt_m, color='blue', linestyle='--', linewidth=2, label=f'GT ({gt_m:.3f} m)')
        
    ax1.set_title("1. Vista Temporal (Evolución de medidas, AR y Ángulo)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Longitud (m)", fontweight='bold')
    ax1.tick_params(labelbottom=False) # Ocultar números X
    ax1.legend(loc='lower right', fontsize=10)

    # PANEL B: Aspect Ratio vs Tiempo
    ax_ar = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax_ar.plot(valid_df['frame_num'], valid_df['aspect_ratio'], marker='^', color='purple', linewidth=1.5)
    ax_ar.axhline(y=ar_thr, color='black', linestyle=':', linewidth=2, label=f'Umbral AR ({ar_thr})')
    ax_ar.set_ylabel("Aspect Ratio", fontweight='bold')
    ax_ar.tick_params(labelbottom=False) # Ocultar números X
    ax_ar.legend(loc='upper right', fontsize=9)

    # PANEL C: Ángulo Z vs Tiempo
    ax_ang = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax_ang.plot(valid_df['frame_num'], valid_df['elevation_deg'], marker='v', color='brown', linewidth=1.5)
    ax_ang.axhline(y=angle_thr, color='black', linestyle=':', linewidth=2, label=f'Umbral Ángulo (±{angle_thr}º)')
    ax_ang.axhline(y=-angle_thr, color='black', linestyle=':', linewidth=2)
    ax_ang.set_ylabel("Ángulo Z (º)", fontweight='bold')
    ax_ang.set_xlabel("Número de Frame", fontweight='bold')
    ax_ang.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_ang.legend(loc='upper right', fontsize=9)


    # ==========================================
    # COLUMNA DERECHA: VISTA ALGORÍTMICA (Ocupa las 3 filas)
    # ==========================================
    ax2 = fig.add_subplot(gs[:, 1]) # El ':' significa "ocupa todas las filas"
    sorted_df['rank'] = np.arange(1, len(sorted_df) + 1)
    
    sns.scatterplot(data=sorted_df, x='rank', y=length_col, hue='status', style='status', 
                    palette=color_dict, markers=markers_dict, s=150, ax=ax2)
    
    ax2.plot(sorted_df['rank'], sorted_df[length_col], color='gray', alpha=0.3, zorder=0)
    ax2.axhline(y=final_length, color='#2ca02c', linestyle='-', linewidth=2.5)
    
    if gt_m is not None:
        ax2.axhline(y=gt_m, color='blue', linestyle='--', linewidth=2)
        
    ax2.set_title("2. Vista Algorítmica (Filtro por Orden Descendente)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Ranking de Medida (1 = La más grande)", fontweight='bold')
    ax2.set_ylabel("") # Quitamos el label Y para que no moleste
    
    # Ponemos la leyenda de los estados aquí, donde hay más espacio
    ax2.legend(loc='upper right', fontsize=10, title="Estado del Frame en el Algoritmo")
    
    # Sincronizamos los límites del eje Y del panel algorítmico con el panel de longitud
    ax2.set_ylim(ax1.get_ylim())
    
    plt.tight_layout()
    plt.show()

# EJEMPLO DE USO: 
# plot_smart_filter_explanation_pro(df_raw, "12_07_31_0", 1)


def plot_thresholds_interaction(df_base, ar_range=None, angles_to_test=None, optimal_ar=3):
    """
    Evaluates and plots the interaction between Aspect Ratio and Z-Angle thresholds.
    It shows how the combination of both affects the absolute error and frame retention.
    
    Returns:
    --------
    df_interaction: pd.DataFrame with the numerical results of the evaluation.
    """

    # Default values if not provided as arguments
    if ar_range is None:
        ar_range = np.arange(1.0, 5, 0.2)
    if angles_to_test is None:
        angles_to_test = [15, 20, 25, 30,60, 90]
        
    def evaluate_thresholds(df, ar_thr, angle_thr):
        # Filter by the given Aspect Ratio and Angle thresholds
        df_filt = df[(df['aspect_ratio'] >= ar_thr) & (df['elevation_deg'].abs() <= angle_thr)]
        if df_filt.empty: return np.nan, 0, 0
        
        # Group by day, folder, and track_id to prevent mixing tracks from different days
        df_agg = df_filt.groupby(['video_day', 'source_folder', 'track_id']).apply(smart_aggregator).dropna().reset_index()
        if df_agg.empty: return np.nan, 0, 0
        
        # Calculate metrics
        mean_err = df_agg['abs_error_cm'].mean()
        retained_frames = len(df_filt)
        retained_tracks = len(df_agg)
        
        return mean_err, retained_frames, retained_tracks

    # =====================================================================
    # 3. GATHER COMBINED DATA
    # =====================================================================
    print("Calculating interaction between Aspect Ratio and Angle...")
    results = []

    for ang in tqdm(angles_to_test, desc="Evaluating Angles"):
        for ar in ar_range:
            err, frames, tracks = evaluate_thresholds(df_base, ar_thr=ar, angle_thr=ang)
            
            # Label for the legend
            if ang == 90:
                ang_label = "No Z-Filter (90º)"
            else:
                ang_label = f"Angle <= {ang}º"
                
            results.append({
                'Angle_Thr': ang_label,
                'AR_Thr': ar,
                'Error_cm': err,
                'Frames': frames,
                'Tracks': tracks
            })

    df_interaction = pd.DataFrame(results)

    # =====================================================================
    # 4. PLOT INTERACTION (PAPER-READY)
    # =====================================================================
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Color palette: ensuring good contrast between lines
    palette = sns.color_palette("Set1", n_colors=len(angles_to_test))

    # --- LEFT PANEL: ABSOLUTE ERROR ---
    sns.lineplot(data=df_interaction, x='AR_Thr', y='Error_cm', hue='Angle_Thr', 
                 marker='o', markersize=8, linewidth=2.5, palette=palette, ax=axes[0])

    axes[0].axvline(x=optimal_ar, color='black', linestyle=':', linewidth=2, label=f'Chosen AR Threshold ({optimal_ar})')
    axes[0].set_title("Interaction: Angle Effect on Error by Aspect Ratio", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Minimum Aspect Ratio Threshold (AR)", fontweight='bold')
    axes[0].set_ylabel("Mean Absolute Error (cm)", fontweight='bold')
    axes[0].legend(title="Pitch Restriction")

    # --- RIGHT PANEL: RETAINED FRAMES ---
    sns.lineplot(data=df_interaction, x='AR_Thr', y='Frames', hue='Angle_Thr', 
                 marker='s', markersize=8, linewidth=2.5, palette=palette, ax=axes[1])

    axes[1].axvline(x=optimal_ar, color='black', linestyle=':', linewidth=2, label=f'Chosen AR Threshold ({optimal_ar})')
    axes[1].set_title("Data Cost: Retained frames when combining filters", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Minimum Aspect Ratio Threshold (AR)", fontweight='bold')
    axes[1].set_ylabel("Number of Useful Retained Frames", fontweight='bold')
    axes[1].legend(title="Pitch Restriction")

    plt.tight_layout()
    plt.show()
    
    return df_interaction

def plot_track_evolution_detailed(df, folder_code, track_id, ar_thr=3, angle_thr=20.0):
    """
    Plots the temporal evolution of the measurement, aspect ratio, and pitch angle.
    Shades frames in red that do not meet the quality criteria and marks frames
    where the 3D measurement failed (length = -1) with a giant red X.
    """
    folder_str = str(folder_code)
    track_str = str(track_id)
    
    # Filter data for the specific video and track
    mask = (df['source_folder'].astype(str) == folder_str) & (df['track_id'].astype(str) == track_str)
    fish_data = df[mask].copy()
    
    if fish_data.empty:
        print(f"❌ ERROR: No data found for video '{folder_str}' and track '{track_str}'")
        return
        
    # Extract numerical frame ID for proper chronological sorting
    fish_data['frame_num'] = fish_data['frame_id'].astype(str).str.extract(r'(\d+)').astype(float).astype(int)
    fish_data = fish_data.sort_values(by='frame_num')
    
    gt_cm = fish_data['gt'].iloc[0]
    gt_m = gt_cm / 100.0 if gt_cm > 0 else None
    
    # Identify bad frames based on the thresholds
    is_3d_ok = fish_data['is_3D_complete'].fillna(False).astype(bool)
    is_ar_ok = fish_data['aspect_ratio'].fillna(0) >= ar_thr
    is_angle_ok = fish_data['elevation_deg'].fillna(90).abs() <= angle_thr
    
    perfect_mask = is_3d_ok & is_ar_ok & is_angle_ok
    bad_frames = fish_data[~perfect_mask]['frame_num']
    
    # --- CREATE FIGURE WITH 3 PANELS ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    fig.suptitle(f"Kinematic Analysis and Measurement - [Video: {folder_str} | Track ID: {track_str}]", fontsize=15, fontweight='bold', y=0.96)
    
    # ==========================================
    # PANEL 1: LENGTH 
    # ==========================================
    ax1 = axes[0]
    
    # 1. Detect where the measurement failed BEFORE cleaning it (-1 means 3D failure)
    measurement_fail_frames = fish_data[fish_data['filtered_length'] == -1]['frame_num']
    
    # 2. Hide the -1s from the continuous line to avoid breaking the Y-axis scale
    fish_data['raw_length'] = fish_data['raw_length'].replace(-1, np.nan)
    fish_data['filtered_length'] = fish_data['filtered_length'].replace(-1, np.nan)

    # Draw valid measurement lines
    ax1.plot(fish_data['frame_num'], fish_data['raw_length'], marker='o', color='orange', alpha=0.6, label='Raw Length 3D')
    ax1.plot(fish_data['frame_num'], fish_data['filtered_length'], marker='s', color='blue', alpha=0.8, label='Filtered Length (HDBSCAN)')
    
    # 3. Draw failures (-1) as red crosses at the bottom of the plot
    if not measurement_fail_frames.empty:
        ax1.scatter(measurement_fail_frames, [0.02] * len(measurement_fail_frames), 
                    color='red', marker='X', s=100, zorder=5, label='3D Measurement Failure (-1)')

    if gt_m is not None:
        ax1.axhline(y=gt_m, color='green', linestyle='--', linewidth=2.5, label=f'Ground Truth ({gt_m:.3f} m)')
    
    # Dynamic Y-Limit (Adapts if there are huge fish or massive errors)
    max_measured = fish_data[['raw_length', 'filtered_length']].max().max()
    upper_limit = max(0.45, max_measured + 0.05) if pd.notna(max_measured) else 0.45
    ax1.set_ylim(0.0, upper_limit)
    
    # Detailed grid 
    ax1.yaxis.set_major_locator(MultipleLocator(0.1))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax1.grid(True, which='major', linestyle='-', alpha=0.7)
    ax1.grid(True, which='minor', linestyle=':', alpha=0.4)
    
    ax1.set_ylabel("Measured Length (m)", fontweight='bold')
    ax1.legend(loc='upper right')
    
    # ==========================================
    # PANEL 2: ASPECT RATIO
    # ==========================================
    ax2 = axes[1]
    ax2.plot(fish_data['frame_num'], fish_data['aspect_ratio'], marker='^', color='purple', linewidth=2)
    ax2.axhline(y=ar_thr, color='black', linestyle=':', linewidth=2, label=f'AR Threshold ({ar_thr})')
    
    ax2.set_ylabel("Aspect Ratio", fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')
    
    # ==========================================
    # PANEL 3: ELEVATION ANGLE (PITCH)
    # ==========================================
    ax3 = axes[2]
    ax3.plot(fish_data['frame_num'], fish_data['elevation_deg'], marker='v', color='brown', linewidth=2)
    ax3.axhline(y=angle_thr, color='black', linestyle=':', linewidth=2, label=f'Angle Threshold (±{angle_thr}º)')
    ax3.axhline(y=-angle_thr, color='black', linestyle=':', linewidth=2)
    
    ax3.set_ylabel("Z-Angle (Pitch) (º)", fontweight='bold')
    ax3.set_xlabel("Frame Number", fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='upper right')
    
    # ==========================================
    # SHADE BAD FRAMES ON ALL 3 PANELS
    # ==========================================
    for ax in axes:
        for bf in bad_frames:
            # Soft red shadow to indicate the frame is discarded
            ax.axvspan(bf - 0.5, bf + 0.5, color='red', alpha=0.15)
            
    # Force integers on the shared X-axis (We don't want frame 2.5)
    axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.show()