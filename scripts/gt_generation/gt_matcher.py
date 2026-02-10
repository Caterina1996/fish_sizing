import pandas as pd

# Función auxiliar para calcular Intersection over Union
def calculate_iou(boxA, boxB):
    # box format: [x, y, w, h]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def script_3_evaluate_new_model():
    # 1. Cargar Ground Truth (Tu verdad absoluta)
    gt = pd.read_csv('ground_truth_master.csv')
    
    # 2. Cargar inferencias del NUEVO modelo (formato similar)
    preds = pd.read_csv('new_model_results.txt') 
    
    # Diccionario con medidas reales (tu metadata)
    medidas_reales = {"pez_a": 25.5, "pez_b": 18.2, "pez_c": 30.0}
    
    results = []

    # Iterar por frames
    unique_frames = preds['frame'].unique()
    
    for frame in unique_frames:
        gt_in_frame = gt[gt['frame'] == frame]
        preds_in_frame = preds[preds['frame'] == frame]
        
        for _, pred_row in preds_in_frame.iterrows():
            pred_box = [pred_row['bb_left'], pred_row['bb_top'], pred_row['width'], pred_row['height']]
            
            best_iou = 0
            matched_fish = None
            
            # Buscar coincidencia en el GT
            for _, gt_row in gt_in_frame.iterrows():
                gt_box = [gt_row['bb_left'], gt_row['bb_top'], gt_row['width'], gt_row['height']]
                iou = calculate_iou(pred_box, gt_box)
                
                if iou > best_iou:
                    best_iou = iou
                    matched_fish = gt_row['id'] # "pez_a"
            
            # Si hay suficiente solapamiento, asumimos que es ese pez
            if best_iou > 0.5: # Umbral estándar
                real_length = medidas_reales.get(matched_fish)
                # Aquí podrías comparar tu 'pred_length' (si tu modelo mide) vs 'real_length'
                results.append({
                    "frame": frame,
                    "predicted_box_id": pred_row['id'],
                    "matched_gt_fish": matched_fish,
                    "real_length": real_length,
                    "iou": best_iou
                })

    df_res = pd.DataFrame(results)
    print(df_res.head())
    print("Mapeo completado. Ahora puedes calcular el error de medición.")

if __name__ == "__main__":
    script_3_evaluate_new_model()
    