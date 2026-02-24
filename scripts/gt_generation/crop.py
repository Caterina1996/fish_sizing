import os
import glob
from PIL import Image

# 1. Definir el patrón de búsqueda y la carpeta de salida
patron = "L1_2024_11_27-13_15_03-*"
carpeta_salida = "//home/slimbook/fish_sizing/out/overfitting_dataset/"

# Crear carpeta de salida si no existe
if not os.path.exists(carpeta_salida):
    os.makedirs(carpeta_salida)

# 2. Buscar las imágenes
archivos = glob.glob(os.path.join(carpeta_salida,patron))
print(f"Se encontraron {len(archivos)} imágenes para procesar.")

for archivo in archivos:
    try:
        with Image.open(archivo) as img:
            width, height = img.size
            
            # 3. Definir el área de recorte
            # Queremos eliminar el tercio superior, así que el corte empieza en height/3
            # La tupla es (izquierda, arriba, derecha, abajo)
            nuevo_top = height // 4
            area_recorte = (0, nuevo_top, width, height)
            
            img_recortada = img.crop(area_recorte)
            
            # 4. Guardar en la carpeta nueva
            nombre_archivo = os.path.basename(archivo)
            ruta_guardado = os.path.join(carpeta_salida, nombre_archivo)
            
            # Guardamos con la misma calidad o formato
            img_recortada.save(ruta_guardado)
            print(f"Procesada: {nombre_archivo}")
            
    except Exception as e:
        print(f"Error procesando {archivo}: {e}")

print("¡Proceso terminado!")