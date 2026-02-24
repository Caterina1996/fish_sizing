import os
from pathlib import Path
from termcolor import cprint

# --- CONFIGURACIÓN ---
# Pon aquí la ruta a tu carpeta de labels que has descargado de Roboflow
LABELS_DIR = "/home/slimbook/Descargas/Pool.v3-overlapping_dataset.yolov11/train/labels" 

def main():
    labels_path = Path(LABELS_DIR)
    
    if not labels_path.exists():
        cprint(f"❌ La ruta no existe: {labels_path}", "red")
        return

    archivos_modificados = 0
    etiquetas_eliminadas = 0

    cprint(f"🧹 Buscando etiquetas duplicadas en {labels_path}...", "cyan")

    for txt_file in labels_path.rglob("*.txt"):
        with open(txt_file, "r") as f:
            lines = f.readlines()

        if not lines:
            continue

        # Usar un set elimina automáticamente las líneas idénticas
        # Mantenemos el orden original (opcional, pero limpio) con un pequeño truco
        unique_lines = list(dict.fromkeys(lines))

        duplicados = len(lines) - len(unique_lines)

        if duplicados > 0:
            # Reescribimos el archivo solo con las líneas únicas
            with open(txt_file, "w") as f:
                f.writelines(unique_lines)
            
            archivos_modificados += 1
            etiquetas_eliminadas += duplicados
            print(f"Corregido {txt_file.name}: -{duplicados} etiquetas fantasma.")

    cprint("\n✅ ¡Limpieza completada!", "green")
    cprint(f"Archivos corregidos: {archivos_modificados}", "yellow")
    cprint(f"Etiquetas duplicadas eliminadas: {etiquetas_eliminadas}", "yellow")

if __name__ == "__main__":
    main()