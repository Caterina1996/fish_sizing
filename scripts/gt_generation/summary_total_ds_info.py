import pandas as pd

# 1. Cargar el CSV (Asegúrate de que la ruta es correcta)
archivo_csv = "/home/slimbook/fish_sizing/scripts/gt_generation/Dataset_Summary_Dataset 1.csv" # Pon la ruta real si esta falla

# Leemos el CSV intentando que detecte el separador automáticamente
df = pd.read_csv(archivo_csv, sep=None, engine='python')

# 2. LIMPIEZA A PRUEBA DE BALAS: Convertir todo lo que tenga coma a número con punto
columnas_numericas = ["Duration_Sec", "Num_Images_Left", "Num_Images_Right", "Size_GB", "Frame_Rate_FPS"]

for col in columnas_numericas:
    if col in df.columns:
        # Lo pasamos a texto, cambiamos la coma por punto y lo forzamos a float (número decimal)
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

# 3. Calcular los totales
totales = {
    "Total_Duration_Sec": df["Duration_Sec"].sum(),
    "Total_Num_Images_Left": df["Num_Images_Left"].sum(),
    "Total_Num_Images_Right": df["Num_Images_Right"].sum(),
    "Total_Size_GB": df["Size_GB"].sum(),
    "Average_Frame_Rate_FPS": df["Frame_Rate_FPS"].mean()
}

# 4. Mostrar los resultados por pantalla
print("\n" + "="*40)
print("📊 RESUMEN DE TOTALES")
print("="*40)
print(f"Duración Total (segundos): {totales['Total_Duration_Sec']:.2f}")
print(f"Total Imágenes (Izquierda): {int(totales['Total_Num_Images_Left'])}")
print(f"Total Imágenes (Derecha):   {int(totales['Total_Num_Images_Right'])}")
print(f"Tamaño Total (GB):         {totales['Total_Size_GB']:.2f}")
print(f"Media de FPS:              {totales['Average_Frame_Rate_FPS']:.2f}")
print("="*40 + "\n")