
import os
from pathlib import Path

# Cambia esto por la ruta a tu carpeta principal (ej: '/ruta/a/2024_11_28')
ROOT_DIR = Path("//home/slimbook/fish_sizing/ARTICLE/DATASET_1/2025_05_08/multiple_fish/") 
day_code = ROOT_DIR.name # Extrae automáticamente '2024_11_28'

# Busca todos los .csv dentro de cualquier carpeta llamada 'corrected_results'
for csv_path in ROOT_DIR.rglob("corrected_results/*.csv"):
    old_name = csv_path.name
    
    # 1. Quitar 'pending_'
    new_name = old_name.replace("corrected_results", "results")
    
    # 2. Añadir el día si no lo tiene ya
    if not new_name.startswith(day_code):
        new_name = f"{day_code}_{new_name}"
        
    # 3. Renombrar el archivo físicamente
    new_path = csv_path.parent / new_name
    if csv_path != new_path:
        csv_path.rename(new_path)
        print(f"✅ Renombrado: {old_name}  -->  {new_name}")

print("🎉 ¡Todos los archivos han sido renombrados!")
