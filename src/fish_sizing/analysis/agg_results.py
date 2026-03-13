#!/usr/bin/env python3
import os
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime  # <-- NUEVA IMPORTACIÓN

def aggregate_csvs(base_dir, output_file, target_csv_name, required_parent=None):
    base_path = Path(base_dir).resolve()
    all_dfs = []

    # 1. Definimos la fecha límite (Hoy a las 00:00:00)
    today_date = datetime.now().date()

    print(f"🔍 Buscando '{target_csv_name}' dentro de carpetas "
          f"'processing_new_stereo_params/results/' en:\n   {base_path}\n")
    print(f"🕒 Ignorando archivos modificados antes de: {today_date}\n")
    
    for root, dirs, files in os.walk(base_path):
        
        dirs[:] = [d for d in dirs if d not in {"_inferred", "debug"}]

        # Si se ha definido filtro de carpeta obligatoria
        if required_parent:
            if required_parent not in Path(root).parts:
                continue

        if os.path.basename(root) == "results" and target_csv_name in files:

            filepath = Path(root) / target_csv_name

            # --- NUEVA LÓGICA DE FILTRADO POR FECHA ---
            mtime = filepath.stat().st_mtime
            file_date = datetime.fromtimestamp(mtime).date()
            
            # Si el archivo es de ayer (o más antiguo), nos lo saltamos
            # if file_date < today_date:
            #     print(f"⏭️ Saltando (Viejo - {file_date}): {filepath.parent.parent.name}")
            #     continue
            # -------------------------------------------

            try:
                df = pd.read_csv(filepath)

                if df.empty:
                    continue

                # Carpeta padre de 'results'
                folder_path = filepath.parent.parent

                try:
                    rel_path = folder_path.relative_to(base_path)
                except ValueError:
                    rel_path = folder_path.name

                df.insert(0, 'source_folder', str(rel_path))

                all_dfs.append(df)
                print(f"✅ Añadido (Nuevo - {file_date}): {rel_path}")

            except Exception as e:
                print(f"❌ Error leyendo {filepath}: {e}")

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv(output_file, index=False)

        print(f"\n🎉 ¡Éxito! Se han agrupado {len(all_dfs)} archivos CSV nuevos.")
        print(f"💾 Archivo final guardado en: {output_file}")
    else:
        print("\n⚠️ No se encontraron archivos válidos nuevos (o todos eran viejos).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Agrupa múltiples CSVs dentro de processing_new_stereo_params/**/results/ ignorando los viejos."
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/slimbook/results_fish_sizing/2024_11_28/multiples_peixos/",
        help="Carpeta raíz donde buscar"
    )
    
    parser.add_argument(
        "--required_parent",
        type=str,
        default=None,
        help="Nombre de carpeta que debe estar en la ruta (ej: processing_new_stereo_params)"
    )

    parser.add_argument(
        "--target_csv",
        type=str,
        default="resume_filtered_smart_angle_ok.csv",
        help="Nombre exacto del CSV a buscar"
    )

    args = parser.parse_args()

    output_name = args.target_csv.replace(".csv", "_aggregated.csv")
    output_path = os.path.join(args.base_dir, output_name)

    aggregate_csvs(
        args.base_dir,
        output_path,
        args.target_csv,
        args.required_parent
    )