#!/usr/bin/env python3.9
import os
import shutil
import argparse
from pathlib import Path

def copy_results_locally(src_dir, dest_dir):
    src_path = Path(src_dir).resolve()
    dest_path = Path(dest_dir).resolve()

    print(f"🔍 Buscando CSVs en: {src_path}")
    print(f"📁 Destino local: {dest_path}\n")

    copied_files = 0

    for root, dirs, files in os.walk(src_path):
            # Optimización extrema: saltamos carpetas pesadas
            dirs[:] = [d for d in dirs if not d.startswith("frame_") and d not in {"_inferred", "debug", "original_images"}]

            if os.path.basename(root) == "results":
                # SOLO buscamos el archivo maestro
                target_file = "all_fish_info_raw.csv"
                
                if target_file in files:
                    src_file = Path(root) / target_file

                    try:
                        rel_path = src_file.relative_to(src_path)
                    except ValueError:
                        continue

                    # Extraer código de la carpeta (ej: 13_44_55)
                    parent_folder = src_file.parent.parent.name
                    grandparent_folder = src_file.parent.parent.parent.name
                    
                    if parent_folder in ["compressed", "0", "1", "results_article_basic"]:
                        folder_code = f"{grandparent_folder}_{parent_folder}"
                    else:
                        folder_code = parent_folder

                    # Renombramos para no tener 50 archivos llamados igual
                    new_filename = f"{folder_code}_raw.csv"

                    dest_file = dest_path / rel_path.parent / new_filename
                    dest_file.parent.mkdir(parents=True, exist_ok=True)

                    shutil.copy2(src_file, dest_file)
                    copied_files += 1

                    print(f"✅ Extraído: {rel_path.parent} -> {new_filename}")

    print("-" * 50)
    print(f"🎉 ¡Extracción limpia! Se copiaron {copied_files} archivos RAW.")
    print(f"💾 Listos para Jupyter en: {dest_path}")

def main():
    parser = argparse.ArgumentParser(description="Exporta todos los CSVs de results a una carpeta local conservando la estructura.")
    parser.add_argument("--src", "-s", default="/media/slimbook/easystore/results_fish_sizing/seleccio_article/lanty1/2025_08_21/1_peix/", help="Ruta base del disco externo")
    parser.add_argument("--dest", "-d", default="/home/slimbook/fish_sizing/ARTICLE/", help="Ruta de la carpeta local de destino")
    args = parser.parse_args()

    if not os.path.exists(args.src):
        print(f"❌ Error: La ruta de origen no existe: {args.src}")
        return

    copy_results_locally(args.src, args.dest)

if __name__ == "__main__":
    main()