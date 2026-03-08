
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

        root_path = Path(root)

        # Buscamos la carpeta 'results' cuyo padre sea 'results_article_basic_nou'
        # if root_path.name == "corrected_results" and root_path.parent.name == "":
        if root_path.name == "corrected_results":
            
            target_file = "all_fish_info_raw.csv"
            
            if target_file in files:
                src_file = root_path / target_file

                try:
                    rel_path = src_file.relative_to(src_path)
                except ValueError:
                    continue

                # rel_path será algo como: 2024_11_12/10_44_11/0/results_article_basic_nou/results/all_fish_info_raw.csv
                # rel_path.parents[0] -> .../results
                # rel_path.parents[1] -> .../results_article_basic_nou
                # rel_path.parents[2] -> 2024_11_12/10_44_11/0 (¡Esto es lo que queremos!)
                
                base_path_for_name = rel_path.parents[2]
                
                # Si el archivo estuviera justo en la raíz (raro), base_path_for_name sería "."
                # Lo gestionamos para evitar nombres extraños
                if str(base_path_for_name) == ".":
                    folder_code = "root"
                else:
                    # Convertimos la ruta en texto y cambiamos los separadores ("/" o "\") por "_"
                    # Resultado: "2024_11_12_10_44_11_0"
                    folder_code = str(base_path_for_name).replace(os.sep, "_")

                new_filename = f"{folder_code}_raw.csv"

                # Guardamos conservando la estructura de carpetas:
                dest_file = dest_path / rel_path.parent / new_filename
                
                # NOTA: Si prefieres que TODOS los CSVs se guarden directamente en la carpeta 
                # de destino SIN crear subcarpetas (ya que el nombre ya tiene toda la ruta), 
                # puedes cambiar la línea anterior por esta:
                # dest_file = dest_path / new_filename

                dest_file.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy2(src_file, dest_file)
                copied_files += 1

                print(f"✅ Extraído: {new_filename}")

    print("-" * 50)
    print(f"🎉 ¡Extracción limpia! Se copiaron {copied_files} archivos RAW.")
    print(f"💾 Listos para Jupyter en: {dest_path}")

def main():
    parser = argparse.ArgumentParser(description="Exporta todos los CSVs filtrados a una carpeta local conservando la estructura.")
    parser.add_argument("--src", "-s", default="///media/slimbook/easystore1/results_fish_sizing/seleccio_article/lanty1/2025_08_21/1_peix/", help="Ruta base del disco externo")
    parser.add_argument("--dest", "-d", default="/home/slimbook/fish_sizing/ARTICLE/1_peix/2025_08_21/", help="Ruta de la carpeta local de destino")
    args = parser.parse_args()

    if not os.path.exists(args.src):
        print(f"❌ Error: La ruta de origen no existe: {args.src}")
        return

    copy_results_locally(args.src, args.dest)

if __name__ == "__main__":
    main()