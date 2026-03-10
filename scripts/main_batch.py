import os
import subprocess

# Configuración del lote
PARENT_BAGS_DIR = "/home/slimbook/bagfiles/campaña_2025" # Ruta Host
OUTPUT_BASE_DIR = "/home/slimbook/fish_sizing/out"      # Ruta Host
STEREO_CFG      = "/home/slimbook/fish_sizing/config/stereo.yaml"
PIPE_PARAMS     = "/home/slimbook/fish_sizing/config/pipeline_params.yaml"

def run_batch():
    # Listar todos los archivos .bag
    if not os.path.exists(PARENT_BAGS_DIR):
        print(f"❌ Error: No existe la carpeta de entrada {PARENT_BAGS_DIR}")
        return

    bags = [f for f in os.listdir(PARENT_BAGS_DIR) if f.endswith(".bag")]
    print(f"📂 Encontrados {len(bags)} bagfiles. Iniciando proceso...")

    for bag in bags:
        bag_name = os.path.splitext(bag)[0]
        input_file = os.path.join(PARENT_BAGS_DIR, bag)
        output_folder = os.path.join(OUTPUT_BASE_DIR, bag_name)

        print(f"\n🚀 >>> PROCESANDO: {bag}")
        
        # Llamada al sistema para ejecutar el main_pipeline.py
        # El propio main_pipeline se encargará de hacer el transform_path2docker de cada argumento
        cmd = [
            "python3", "main_pipeline.py",
            "--input", input_file,
            "--output", output_folder,
            "--stereo_config", STEREO_CFG,
            "--pipeline_config", PIPE_PARAMS,
            # "--skip_extraction" # Descomentar para pruebas rápidas si ya extrajiste
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error procesando {bag}: {e}")

if __name__ == "__main__":
    run_batch()