#!/usr/bin/env python3
import os
import pandas as pd
from pathlib import Path
import argparse

def aggregate_csvs(base_dir, output_file, target_csv_name):
    base_path = Path(base_dir).resolve()
    all_dfs = []

    print(f"🔍 Buscando archivos '{target_csv_name}' en:\n   {base_path}\n")

    # rglob busca recursivamente en todas las subcarpetas
    for filepath in base_path.rglob(target_csv_name):
        try:
            # Leer el CSV
            df = pd.read_csv(filepath)
            
            # Si el CSV está vacío (solo cabeceras), lo saltamos
            if df.empty:
                continue

            folder_path = filepath.parent
            
            # Si el archivo está dentro de una carpeta llamada 'results', subimos un nivel más
            if folder_path.name == "results":
                folder_path = folder_path.parent
            
            # Calculamos la ruta relativa respecto a la carpeta base
            try:
                rel_path = folder_path.relative_to(base_path)
            except ValueError:
                # Fallback por si hay algún problema de rutas absolutas extrañas
                rel_path = folder_path.name
            
            # --- AÑADIR LA COLUMNA AL PRINCIPIO ---
            # insert(posición, nombre_columna, valor) -> 0 es la primera columna
            df.insert(0, 'source_folder', str(rel_path))
            
            all_dfs.append(df)
            print(f"✅ Añadido: {rel_path}")
            
        except Exception as e:
            print(f"❌ Error leyendo {filepath}: {e}")

    # --- UNIR Y GUARDAR ---
    if all_dfs:
        # Unir todos los DataFrames apilándolos
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        # Guardar el CSV resultante
        final_df.to_csv(output_file, index=False)
        print(f"\n🎉 ¡Éxito! Se han agrupado {len(all_dfs)} archivos CSV.")
        print(f"💾 Archivo final guardado en: {output_file}")
    else:
        print("\n⚠️ No se encontraron archivos válidos (o estaban todos vacíos) para agrupar.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agrupa múltiples CSVs de resultados en uno solo.")
    
    # Por defecto usará tu ruta, pero puedes cambiarla por consola si quieres
    parser.add_argument("--base_dir", type=str, 
                        default="//media/slimbook/easystore/results_fish_sizing/Peixos_piscina/processed/2024_11_12/",
                        help="Carpeta raíz donde buscar los CSVs")
    
    parser.add_argument("--target_csv", type=str, 
                        default="resume_filtered_smart_angle_ok.csv",
                        help="Nombre exacto del CSV a buscar")
    
    args = parser.parse_args()
    
    # El archivo de salida se llamará igual pero con _aggregated al final, y se guardará en la carpeta raíz
    output_name = args.target_csv.replace(".csv", "_aggregated.csv")
    output_path = os.path.join(args.base_dir, output_name)
    
    aggregate_csvs(args.base_dir, output_path, args.target_csv)