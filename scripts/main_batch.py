import os
import subprocess
import sys
import logging
from pathlib import Path 
import re


from fish_sizing.utils.tools import cprint_and_log, setup_logger
from fish_sizing.utils.config import TOPICS_DICT, PROCESSING_PIPELINES, transform_path2docker

# Configuración del lote
PARENT_BAGS_DIR_0 = "/home/slimbook/bagfiles/peixos_morts_piscina_v3/2024_11_28/multiples_peixos/" 
OUTPUT_BASE_DIR_0 = "/home/slimbook/results_fish_sizing/2024_11_28/multiples_peixos/"      
STEREO_CFG      = "/home/slimbook/fish_sizing/config/stereo_config.yaml"
PIPE_PARAMS     = "/home/slimbook/fish_sizing/config/pipeline_params.yaml"

def run_batch():
    # 1. Setup del log global para el Batch
    
    OUTPUT_BASE_DIR = transform_path2docker(OUTPUT_BASE_DIR_0)
    PARENT_BAGS_DIR = transform_path2docker(PARENT_BAGS_DIR_0)
    
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    setup_logger(OUTPUT_BASE_DIR) 
    
    if not os.path.exists(PARENT_BAGS_DIR):
        cprint_and_log(f"❌ Error: No existe la carpeta de entrada {PARENT_BAGS_DIR}", "red", level=logging.ERROR)
        return

    # 2. BÚSQUEDA RECURSIVA: Encuentra .bag en la carpeta raíz y en TODAS sus subcarpetas
    bags = list(Path(PARENT_BAGS_DIR).rglob("*.bag"))
    cprint_and_log(f"📂 Encontrados {len(bags)} bagfiles en {PARENT_BAGS_DIR} (y subcarpetas)", "cyan")

    for bag_path in bags:
        # Extraemos el nombre sin el '.bag'
        raw_bag_name = bag_path.stem 
        
        # 1. Quitar el prefijo fijo 'stereo_camera_images_'
        prefix = "stereo_camera_images_"
        if raw_bag_name.startswith(prefix):
            name_without_prefix = raw_bag_name[len(prefix):] # Deja: '2024-11-28-13-32-16_0'
        else:
            name_without_prefix = raw_bag_name
            
        # 2. Buscar si empieza por una fecha (YYYY-MM-DD)
        date_match = re.match(r'^(\d{4}-\d{2}-\d{2})-(.*)', name_without_prefix)
        
        if date_match:
            bag_date = date_match.group(1)       # Ej: '2024-11-28'
            rest_of_name = date_match.group(2)   # Ej: '13-32-16_0'
            
            # Convertimos a barras bajas para comparar con las carpetas
            bag_date_underscores = bag_date.replace('-', '_') # Ej: '2024_11_28'
            
            # bag_path.parts contiene todas las carpetas de la ruta una por una
            if bag_date in bag_path.parts or bag_date_underscores in bag_path.parts:
                bag_name = rest_of_name      # La fecha ya está en la carpeta -> Se recorta
            else:
                bag_name = name_without_prefix # No está en la carpeta -> Conservamos la fecha
        else:
            bag_name = name_without_prefix # Fallback si no hay fecha reconocible
        # Mantiene la estructura de carpetas (por si tienes subcarpetas de fechas)
        rel_path = bag_path.relative_to(PARENT_BAGS_DIR).parent
        output_folder = os.path.join(OUTPUT_BASE_DIR, str(rel_path), bag_name)
        
        # 3. CHECK: Tu pipeline guarda los datos en output_folder/results/all_fish_info_raw.csv
        # Si ese archivo existe, asumimos que ya se procesó.
        check_file = os.path.join(output_folder, "results", "all_fish_info_raw.csv")
        
        if os.path.exists(check_file):
            cprint_and_log(f"⏭️  SALTANDO: {bag_name} (Ya existe resultado en {output_folder})", "yellow")
            continue

        # 4. EJECUCIÓN DEL PIPELINE
        cprint_and_log(f"\n🚀 >>> PROCESANDO: {bag_name}", "magenta", attrs=["bold"])
        cprint_and_log(f"   Input: {bag_path}", "magenta")
        cprint_and_log(f"   Output: {output_folder}", "magenta")
        
        # ARGUMENTOS CORREGIDOS para coincidir con main_pipeline.py
        cmd = [
            sys.executable, "main_pipeline.py", # Si estás fuera de Docker, pon 'python3' en vez de sys.executable si te da fallos.
            "--input", str(bag_path),
            "--out_path", output_folder,
            "--stereo_config", STEREO_CFG,
            "--pipeline_config_path", PIPE_PARAMS
        ]

        try:
            # Ejecutamos y esperamos
            subprocess.run(cmd, check=True)
            cprint_and_log(f"✅ FINALIZADO con éxito: {bag_name}\n", "green")
        except subprocess.CalledProcessError:
            cprint_and_log(f"❌ ERROR: El pipeline falló para {bag_name}. Revisar logs de la carpeta.\n", "red", level=logging.ERROR)
            break
        except KeyboardInterrupt:
            cprint_and_log("\n🛑 Proceso por lotes detenido por el usuario.", "yellow", attrs=["bold"])
            break

    cprint_and_log("🏁 **************************************", "magenta", attrs=["bold"])
    cprint_and_log("🏁 Proceso de lote terminado.", "magenta", attrs=["bold"])
    cprint_and_log("🏁 **************************************", "magenta", attrs=["bold"])

if __name__ == "__main__":
    run_batch()