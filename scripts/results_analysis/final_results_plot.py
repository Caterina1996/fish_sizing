import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ===================================================================
# 1. CONFIGURACIÓN Y RUTAS
# ===================================================================

DS1_MASTER_PATH = Path("/home/slimbook/fish_sizing/ARTICLE/DATASET_1/aggregated_data/MASTER_all_fish_all_days.csv")
DS2_MASTER_PATH = Path("/home/slimbook/fish_sizing/ARTICLE/DATASET_2/aggregated_data/MASTER_all_fish_all_days.csv")

# ===================================================================
# 2. CARGA Y FILTRADO DE DATOS
# ===================================================================
def load_and_filter_dataset(path, dataset_name):
    if not path.exists():
        print(f"⚠️ Archivo no encontrado: {path}")
        return pd.DataFrame()
        
    df = pd.read_csv(path)
    df['dataset'] = dataset_name
    
    # 1er Filtro: Eliminar frames con error de Angle o Aspect Ratio 
    # Revisar si s'han de recalcular les condicions!
    df_valid = df[df['failure_reason']=="measured"].copy()
    
    # 2o Filtro: Eliminar tracks con 5 frames o menos
    # Creamos un identificador único para cada pez/track
    df_valid['track_uid'] = df_valid['video_day'].astype(str) + "_" + df_valid['video_name'].astype(str) + "_" + df_valid['track_id'].astype(str)
    
    # Contamos cuántos frames tiene cada track
    track_counts = df_valid['track_uid'].value_counts()
    valid_tracks = track_counts[track_counts > 5].index
    
    # Filtramos conservando solo los tracks largos
    df_filtered = df_valid[df_valid['track_uid'].isin(valid_tracks)].copy()
    
    # Calculamos el error relativo porcentual (opcional pero muy útil para el paper)
    df_filtered['rel_error_perc'] = (df_filtered['abs_error_cm'] / df_filtered['gt']) * 100
    
    print(f"✅ {dataset_name}: {len(df_filtered)} frames válidos retenidos de {len(df)} originales.")
    return df_filtered

df_ds1 = load_and_filter_dataset(DS1_MASTER_PATH, "Dataset_1")
df_ds2 = load_and_filter_dataset(DS2_MASTER_PATH, "Dataset_2")

# Unimos ambos datasets
df_all = pd.concat([df_ds1, df_ds2], ignore_index=True)

# ===================================================================
# 3. CREACIÓN DE LAS CATEGORÍAS PARA EL ANÁLISIS
# ===================================================================
# Vamos a crear un nuevo DataFrame donde asignaremos explícitamente la etiqueta
# de la categoría que quieres para hacer el Boxplot súper fácil.

plot_data = []

def append_category(df_subset, category_name):
    if not df_subset.empty:
        temp = df_subset.copy()
        temp['Analysis_Group'] = category_name
        plot_data.append(temp)

# -- DATOS DATASET 1 --
df1_single = df_all[(df_all['dataset'] == 'Dataset_1') & (df_all['scenario'] == 'single_fish')]
df1_multi  = df_all[(df_all['dataset'] == 'Dataset_1') & (df_all['scenario'] == 'multiple_fish')]
df1_global = df_all[(df_all['dataset'] == 'Dataset_1')]

append_category(df1_single, "DS1\nSingle")
append_category(df1_multi, "DS1\nMultiple")
append_category(df1_global, "DS1\nGlobal")

# -- DATOS DATASET 2 --
df2_single = df_all[(df_all['dataset'] == 'Dataset_2') & (df_all['scenario'] == 'single_fish')]
df2_multi  = df_all[(df_all['dataset'] == 'Dataset_2') & (df_all['scenario'] == 'multiple_fish')]
df2_global = df_all[(df_all['dataset'] == 'Dataset_2')]

append_category(df2_single, "DS2\nSingle")
append_category(df2_multi, "DS2\nMultiple")
append_category(df2_global, "DS2\nGlobal")

# -- DATOS GLOBALES (DS1 + DS2) --
df_glob_single = df_all[df_all['scenario'] == 'single_fish']
df_glob_multi  = df_all[df_all['scenario'] == 'multiple_fish']
df_glob_total  = df_all

append_category(df_glob_single, "Global\nSingle")
append_category(df_glob_multi, "Global\nMultiple")
append_category(df_glob_total, "GLOBAL\nTOTAL")

df_plot = pd.concat(plot_data, ignore_index=True)

# ===================================================================
# 4. TABLA DE MÉTRICAS (PAPER READY)
# ===================================================================
print("\n" + "="*80)
print(f"{'CATEGORÍA':<15} | {'N (Frames)':<10} | {'MAE (cm)':<10} | {'STD (cm)':<10} | {'MAPE (%)':<10}")
print("-" * 80)

metrics_list = []
for group in df_plot['Analysis_Group'].unique():
    subset = df_plot[df_plot['Analysis_Group'] == group]
    
    n_frames = len(subset)
    mae = subset['abs_error_cm'].mean()     # Mean Absolute Error
    std = subset['abs_error_cm'].std()      # Desviación estándar del error absoluto
    mape = subset['rel_error_perc'].mean()  # Mean Absolute Percentage Error
    
    group_clean = group.replace("\n", " ")
    print(f"{group_clean:<15} | {n_frames:<10} | {mae:<10.2f} | {std:<10.2f} | {mape:<10.2f}")

print("="*80 + "\n")

# ===================================================================
# 5. BOXPLOT DE CALIDAD DE PUBLICACIÓN
# ===================================================================
plt.figure(figsize=(14, 7))
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# Definimos una paleta de colores para diferenciar bien
# Grises/Azules para DS1, Naranjas/Rojos para DS2, Verdes/Morados para Globales
palette = [
    "#A9CCE3", "#5499C7", "#2471A3",  # DS1
    "#F5CBA7", "#EB984E", "#CA6F1E",  # DS2
    "#ABEBC6", "#58D68D", "#8E44AD"   # Globales
]

ax = sns.boxplot(
    data=df_plot, 
    x='Analysis_Group', 
    y='abs_error_cm', 
    palette=palette,
    showfliers=False, # Pon True si quieres ver los outliers (los puntitos fuera de los bigotes)
    linewidth=1.5
)

plt.title("Absolute Sizing Error Across Datasets and Scenarios", fontsize=16, fontweight='bold', pad=20)
plt.ylabel("Absolute Error (cm)", fontsize=14, fontweight='bold')
plt.xlabel("", fontsize=14) # Dejamos el eje X sin título porque las categorías ya son claras

# Opcional: Añadir una línea roja discontinua en el objetivo de error (ej: 1 cm)
plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Target Error (1 cm)')
plt.legend()

plt.tight_layout()
plt.savefig("Final_Metrics_Boxplot.png", dpi=300) # Lo guarda a alta resolución
plt.show()