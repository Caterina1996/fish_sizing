import os
import subprocess
import sys
import logging

# Reutilizamos tu herramienta de logging
from fish_sizing.utils.tools import cprint_and_log, setup_logger

# Configuración del lote
PARENT_BAGS_DIR = "/home/slimbook/bagfiles/campaña_2025" 
OUTPUT_BASE_DIR = "/home/slimbook/fish_sizing/out"      
STEREO_CFG      = "/home/slimbook/fish_sizing/config/stereo.yaml"
PIPE_PARAMS     = "/home/slimbook/fish_sizing/config/pipeline_params.yaml"

def run_batch():
    # 1. Setup de un log global para el Batch (opcional)
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    # Podemos crear un log específico para el proceso por lotes
    batch_log_path = os.path.join(OUTPUT_BASE_DIR, "batch_execution.log")
    
    if not os.path.exists(PARENT_BAGS_DIR):
        cprint_and_log(f"❌ Error: No existe la carpeta de entrada {PARENT_BAGS_DIR}", "red", level=logging.ERROR)
        return

    bags = [f for f in os.listdir(PARENT_BAGS_DIR) if f.endswith(".bag")]
    cprint_and_log(f"📂 Encontrados {len(bags)} bagfiles en {PARENT_BAGS_DIR}", "cyan")

    for bag in bags:
        bag_name = os.path.splitext(bag)[0]
        input_file = os.path.join(PARENT_BAGS_DIR, bag)
        output_folder = os.path.join(OUTPUT_BASE_DIR, bag_name)
        
        # 2. CHECK: Skip si ya existe el resultado final
        final_result_file = os.path.join(output_folder, "final_measurements.csv")
        
        if os.path.exists(final_result_file):
            cprint_and_log(f"⏭️  SALTANDO: {bag} (Ya existe resultado en {output_folder})", "yellow")
            continue

        # 3. EJECUCIÓN DEL PIPELINE
        cprint_and_log(f"🚀 >>> PROCESANDO: {bag}", "magenta", attrs=["bold"])
        
        cmd = [
            sys.executable, "main_pipeline.py",
            "--input", input_file,
            "--output", output_folder,
            "--stereo_config", STEREO_CFG,
            "--pipeline_config", PIPE_PARAMS
        ]

        try:
            # Ejecutamos y esperamos
            subprocess.run(cmd, check=True)
            cprint_and_log(f"✅ FINALIZADO con éxito: {bag}", "green")
        except subprocess.CalledProcessError:
            cprint_and_log(f"❌ ERROR: El pipeline falló para {bag}.", "red", level=logging.ERROR)
        except KeyboardInterrupt:
            cprint_and_log("\n🛑 Proceso por lotes detenido por el usuario.", "yellow", attrs=["bold"])
            break

    cprint_and_log("🏁 Proceso de lote terminado.", "magenta", attrs=["bold"])
    cprint_and_log("🏁 **************************************.", "magenta", attrs=["bold"])
    cprint_and_log("🏁 **************************************.", "magenta", attrs=["bold"])
    cprint_and_log("🏁 **************************************.", "magenta", attrs=["bold"])

if __name__ == "__main__":
    run_batch()