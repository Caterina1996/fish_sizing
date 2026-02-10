import json
import pandas as pd

def script_2_create_ground_truth():
    # Cargar mapeo manual
    with open('mapping.json', 'r') as f:
        id_map = json.load(f) # Ejemplo: {"1": "pez_a", "5": "pez_b"}

    # Cargar detecciones crudas
    # Asumimos CSV sin cabecera o con ella, ajusta según Script 1
    df = pd.read_csv('detections_raw.txt') 

    final_rows = []

    for index, row in df.iterrows():
        track_id = str(int(row['id']))
        
        if track_id in id_map:
            real_fish_id = id_map[track_id]
            
            if real_fish_id == "ignore":
                continue # Saltamos basura
            
            # Creamos la fila limpia
            # frame, fish_id (string o int mapeado), box...
            new_row = row.copy()
            new_row['id'] = real_fish_id # Ahora el ID es "pez_a" o un entero único para A
            final_rows.append(new_row)

    # Guardar GT Final
    df_final = pd.DataFrame(final_rows)
    df_final.to_csv('ground_truth_master.csv', index=False)
    print(f"Ground Truth generado con {len(df_final)} instancias etiquetadas.")

if __name__ == "__main__":
    script_2_create_ground_truth()