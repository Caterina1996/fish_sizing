import os
from pathlib import Path

# Cambia esto por la ruta a tu carpeta principal
ROOT_DIR = Path("//home/slimbook/fish_sizing/ARTICLE/DATASET_2/") 


print(f"🔍 Buscando carpetas 'corrected_results' en: {ROOT_DIR}\n")

# Buscamos directamente las CARPETAS llamadas 'corrected_results'
carpetas_a_renombrar = list(ROOT_DIR.rglob("corrected_results"))

if not carpetas_a_renombrar:
    print("⚠️ No se encontró ninguna carpeta llamada 'corrected_results'.")
else:
    for carpeta_corrected in carpetas_a_renombrar:
        if carpeta_corrected.is_dir():
            parent_dir = carpeta_corrected.parent
            carpeta_results = parent_dir / "results"
            carpeta_old_results = parent_dir / "old_results"
            
            # 1. Si existe 'results', la renombramos a 'old_results' primero
            if carpeta_results.exists() and carpeta_results.is_dir():
                if carpeta_old_results.exists():
                    print(f"   ⚠️ AVISO: Ya existe 'old_results' en {parent_dir.name}. No se tocará la carpeta 'results' actual para evitar pérdida de datos.")
                else:
                    carpeta_results.rename(carpeta_old_results)
                    print(f"   📦 Apartado: {parent_dir.name}/results  -->  old_results")
            
            # 2. Ahora que el camino está libre, renombramos 'corrected_results' a 'results'
            if not carpeta_results.exists():
                carpeta_corrected.rename(carpeta_results)
                print(f"   ✅ Renombrado: {parent_dir.name}/corrected_results  -->  results")
            else:
                print(f"   ❌ Bloqueado: No se pudo renombrar 'corrected_results' en {parent_dir.name} porque 'results' sigue ocupando el sitio.")

    print(f"\n🎉 ¡Proceso terminado! Se revisaron {len(carpetas_a_renombrar)} carpetas.")