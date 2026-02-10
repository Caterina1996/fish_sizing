import cv2
import csv

# --- CONFIGURACIÓN ---
VIDEO_PATH = 'video_peces.mp4'
OUTPUT_DETECTIONS = 'detections_raw.txt'
OUTPUT_INDEX = 'track_index_review.txt'

def run_inference_on_frame(frame):
    # AQUÍ VA TU CÓDIGO DE INFERENCIA (YOLO, DeepSort, etc.)
    # Debe devolver una lista de: [x1, y1, w, h, conf, track_id]
    # Ejemplo simulado:
    return [[100, 100, 50, 30, 0.95, 1], [200, 200, 60, 40, 0.90, 5]]

def script_1_inference_and_index():
    cap = cv2.VideoCapture(VIDEO_PATH)
    seen_track_ids = set()
    
    with open(OUTPUT_DETECTIONS, 'w') as f_det, open(OUTPUT_INDEX, 'w') as f_idx:
        # Escribir cabeceras
        f_det.write("frame,id,bb_left,bb_top,width,height,conf\n")
        f_idx.write("track_id,first_appearance_frame,review_status\n")
        
        frame_id = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 1. Ejecutar tu modelo
            detections = run_inference_on_frame(frame)
            
            for det in detections:
                x, y, w, h, conf, track_id = det
                
                # 2. Guardar detección cruda (MOT format simplificado)
                f_det.write(f"{frame_id},{track_id},{x},{y},{w},{h},{conf}\n")
                
                # 3. Detectar si es un track nuevo
                if track_id not in seen_track_ids:
                    seen_track_ids.add(track_id)
                    # Guardamos el chivato: "El track 5 aparece en frame 56"
                    f_idx.write(f"{track_id},{frame_id},PENDING_REVIEW\n")
                    print(f"Nuevo pez detectado: Track ID {track_id} en frame {frame_id}")
            
            frame_id += 1
            
    cap.release()
    print("Inferencia terminada. Revisa 'track_index_review.txt' para etiquetar.")

if __name__ == "__main__":
    script_1_inference_and_index()