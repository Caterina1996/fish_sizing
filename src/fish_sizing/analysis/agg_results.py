import os
import pandas as pd

# -------- CONFIG --------
root_dir = r"/home/slimbook/fish_sizing/out/LIMIA/1peix"
output_file = os.path.join(root_dir, "resume_filtered_smart_angle_ok_aggregated.csv")
# ------------------------

all_dfs = []
files_found = 0

for dirpath, dirnames, filenames in os.walk(root_dir):
    target_file = os.path.join(dirpath, "results", "resume_filtered_smart_angle_ok.csv")
    
    if os.path.exists(target_file):
        try:
            df = pd.read_csv(target_file)
            
            # Nombre de la carpeta padre (la que contiene "results")
            parent_folder = os.path.basename(dirpath)
            df["parent_folder"] = parent_folder
            
            all_dfs.append(df)
            files_found += 1
            print(f"✔ Encontrado: {target_file}")
        
        except Exception as e:
            print(f"⚠ Error leyendo {target_file}: {e}")

if not all_dfs:
    print("No se encontraron archivos.")
    exit()

# Concatenar todos
aggregated_df = pd.concat(all_dfs, ignore_index=True)

# Guardar CSV agregado
aggregated_df.to_csv(output_file, index=False)
print(f"\nArchivo agregado guardado en:\n{output_file}")
print(f"Total de archivos combinados: {files_found}")
print(f"Total de filas: {len(aggregated_df)}")

# -------- Estadísticas de error --------
print("\n===== ERRORES MEDIOS GLOBALES =====")

if "abs_error_m" in aggregated_df.columns:
    mean_abs_error = aggregated_df["abs_error_m"].mean()
    print(f"Mean abs_error_m: {mean_abs_error:.4f} m")

if "rel_error_%" in aggregated_df.columns:
    mean_rel_error = aggregated_df["rel_error_%"].mean()
    print(f"Mean rel_error_%: {mean_rel_error:.2f} %")

# También por carpeta (opcional pero útil)
print("\n===== ERRORES MEDIOS POR CARPETA =====")
grouped = aggregated_df.groupby("parent_folder")[["abs_error_m", "rel_error_%"]].mean()
print(grouped)
