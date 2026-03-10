#!/usr/bin/env python3.9
import os
import sys
import subprocess
import argparse
import yaml
from termcolor import cprint

# --- DOCKER MAPPINGS ---
PATH_MAPPINGS = {
    "/home/slimbook/bagfiles": "/home/rosuser/dataset/bagfiles",
    "/home/slimbook/fish_sizing/out": "/home/rosuser/repo/out",
    "/home/slimbook/models": "/home/rosuser/dataset/models/",
    "/home/slimbook/fish_sizing/config" :"/home/rosuser/repo/config/",
    "/media/slimbook/easystore": "/home/rosuser/easystore"
}

USE_DOCKER = True

def transform_path2docker(path: str) -> str:
    """Transforma una ruta del host a la estructura interna del Docker."""
    if not USE_DOCKER or path is None:
        return path
    for host_path, docker_path in PATH_MAPPINGS.items():
        if host_path in path:
            new_path = path.replace(host_path, docker_path)
            return new_path 
    return path

def find_target_directories(root_dir):
    """
    Busca carpetas que contengan 'original_images'. 
    """
    target_dirs = []
    for current_root, dirs, files in os.walk(root_dir):
        if "original_images" in dirs:
            target_dirs.append(current_root)
            dirs[:] = [] 
            
    return target_dirs

def get_gt_from_config(folder_path, default_gt):
    """Intenta leer el GT del archivo run_config.yaml."""
    config_path = os.path.join(folder_path, "run_config.yaml")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                gt = config_data.get('global_variables', {}).get('gt_ground_truth', default_gt)
                if gt != "" and gt is not None:
                    return float(gt)
        except Exception as e:
            cprint(f"⚠️ No se pudo leer el GT de {config_path}: {e}", "yellow")
            
    return default_gt

def main():
    parser = argparse.ArgumentParser(description="Ejecuta el reprocesamiento en caché para múltiples carpetas.")
    parser.add_argument("--root_dir", "-root", type=str, default="//media/slimbook/easystore/results_fish_sizing/seleccio_article/2024_11_28/multiples_peixos/", help="Carpeta padre que contiene todas las subcarpetas de resultados.")
    parser.add_argument("--script_path", type=str, default="calculate_metrics_from_pc_and_scene.py", help="Ruta al script")
    parser.add_argument("--default_gt", type=float, default=33.5, help="Ground Truth por defecto.")
    parser.add_argument("--out_folder_name", type=str, default="to_paper_nou", help="Subcarpeta donde se guardarán los resultados.")
    
    args = parser.parse_args()

    args.root_dir = transform_path2docker(args.root_dir)
    args.script_path = transform_path2docker(args.script_path)

    if not os.path.exists(args.root_dir):
        cprint(f"❌ Error: La carpeta padre {args.root_dir} no existe dentro del contenedor.", "red")
        sys.exit(1)

    if not os.path.exists(args.script_path):
        cprint(f"❌ Error: No se encuentra el script a ejecutar en {args.script_path}.", "red")
        sys.exit(1)

    cprint(f"🔍 Escaneando {args.root_dir} buscando directorios raíz...", "cyan")
    target_folders = find_target_directories(args.root_dir)

    if not target_folders:
        cprint(f"⚠️ No se encontraron carpetas con 'original_images' dentro de {args.root_dir}", "yellow")
        sys.exit(0)

    cprint(f"📋 Encontradas {len(target_folders)} carpetas maestras para reprocesar.\n", "green")

    for i, folder in enumerate(target_folders, 1):
        cprint("\n" + "="*80, "magenta")
        cprint(f"🚀 PROCESANDO CARPETA {i}/{len(target_folders)}", "magenta", attrs=['bold'])
        cprint(f"📁 Origen: {folder}", "magenta")
        
        current_gt = get_gt_from_config(folder, args.default_gt)
        cprint(f"📏 Ground Truth a usar: {current_gt} cm", "cyan")
        
        # --- DEFINIMOS EL DIRECTORIO DE SALIDA ---
        current_out_dir = os.path.join(folder, args.out_folder_name)
        cprint(f"📁 Destino: {current_out_dir}", "cyan")
        
        cmd = [
            sys.executable, 
            args.script_path,
            "-in", folder,
            "--out_dir", current_out_dir,  # PASAMOS EL DIRECTORIO AL SUB-SCRIPT
            "--gt", str(current_gt)
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            cprint(f"\n❌ Error ejecutando el reprocesado en la carpeta {folder}", "red")
            cprint(f"Continuando con la siguiente...\n", "yellow")
            continue
            
    cprint("\n✅ ¡REPROCESAMIENTO POR LOTES COMPLETADO!", "green")

if __name__ == "__main__":
    main()