#!/usr/bin/env python3.9
import argparse
from datetime import datetime, timedelta
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Borra archivos CSV específicos modificados antes de ayer a las 17:00.")
    parser.add_argument("--root_dir", "-root", required=True, help="Carpeta raíz donde buscar.")
    parser.add_argument("--dry_run", action="store_true", default= True, help="Solo muestra qué se borraría sin tocar los archivos.")
    args = parser.parse_args()

    # 1. Calcular el timestamp límite (ayer a las 17:00)
    hoy = datetime.now()
    ayer = hoy - timedelta(days=1)
    fecha_limite = ayer.replace(hour=17, minute=0, second=0, microsecond=0)
    timestamp_limite = fecha_limite.timestamp()

    print(f"🕒 Fecha límite configurada: {fecha_limite}")

    # Archivo objetivo (puedes añadir más a esta tupla si necesitas, ej: all_angle_ok_fish.csv)
    target_files = ("resume_filtered_smart_angle_ok.csv",)
    
    archivos_afectados = 0

    # 2. Búsqueda y evaluación
    print(f"🔍 Escaneando en: {args.root_dir}\n")
    
    for target in target_files:
        for file_path in Path(args.root_dir).rglob(target):
            # Obtener fecha de modificación
            mtime = file_path.stat().st_mtime
            
            if mtime < timestamp_limite:
                fecha_mod = datetime.fromtimestamp(mtime)
                
                if args.dry_run:
                    print(f"[DRY RUN] Borraría: {file_path} \n   -> Modificado: {fecha_mod}")
                else:
                    print(f"🗑️ Borrando: {file_path}")
                    # file_path.unlink() # Borrado real
                    pass
                    
                archivos_afectados += 1

    # 3. Resumen
    print("-" * 50)
    if args.dry_run:
        print(f"✅ Simulacro terminado. Se borrarían {archivos_afectados} archivos.")
        print("Ejecuta de nuevo SIN --dry_run para borrarlos definitivamente.")
    else:
        print(f"✅ Limpieza terminada. {archivos_afectados} archivos borrados.")

if __name__ == "__main__":
    main()