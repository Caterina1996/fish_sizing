import cv2
import numpy as np
import open3d as o3d
import sys
import os
import cloudpickle as pickle
from termcolor import cprint

from fish_detector.detection.fish_detector import FishDetector
from ish_detector.detection.fish2D import Fish2D, FrameScene 

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

IMG_LEFT_PATH = '/home/rosuser/repo/src/out/LIMA/test_cat/frame625_left_image.jpg'
IMG_RIGHT_PATH = '/home/rosuser/repo/src/out/LIMA/test_cat/frame625_right_image.jpg'
MODEL_PATH = "/home/rosuser/repo/dataset/models/yv11l/ylarge_dfishthings_d18_poolv2r_lantytr_nocturnes/weights/best.pt"
OUTPUT_DIR = "/home/rosuser/repo/src/out/LIMA/test_cat/processed_output/"

# Parámetros Visión
USE_CLAHE = True
CLAHE_CLIP = 3.0; CLAHE_GRID = 8 
MIN_DISP = 0; NUM_DISP = 96; BLOCK_SIZE = 5
UNIQUENESS = 10; SPECKLE_WIN = 150; SPECKLE_RNG = 32
WLS_LAMBDA = 8000.0; WLS_SIGMA = 0.8

# Calibración (Decimada x2)
FOCAL_DECIMATED = 730.35 
BASELINE = 0.1203 
CX = 520.70; CY = 380.34 
MAX_DEPTH_METERS = 4.0
min_valid_disparity = (FOCAL_DECIMATED * BASELINE) / MAX_DEPTH_METERS

# ==========================================
# 2. FUNCIONES AUXILIARES
# ==========================================
def get_decimated_Q():
    return np.float32([
        [1, 0, 0, -CX],
        [0, 1, 0, -CY],
        [0, 0, 0, FOCAL_DECIMATED],
        [0, 0, 1.0/BASELINE, 0]  # Positivo
    ])

def match_histogram_stats(source, reference):
    m_src, s_src = cv2.meanStdDev(source)
    m_ref, s_ref = cv2.meanStdDev(reference)
    if s_src[0][0] < 1e-3: return source
    ratio = s_ref[0][0] / s_src[0][0]
    res = (source.astype(np.float32) - m_src[0][0]) * ratio + m_ref[0][0]
    return np.clip(res, 0, 255).astype(np.uint8)

def merge_strips(bboxes, img_h, margin=50):
    """
    Args:
        bboxes (list):  list of [x1,y1,x2,y2]
        img_h (_type_): image height
        margin (int, optional): margin to add to the cropped section Defaults to 50.

    Returns:
        merged (list): lista de franjas que no solapan en forma de [[ystart, yend]] 
    """
    if not bboxes: return []
    intervals = []
    for box in bboxes:
        y1, y2 = int(box[1]), int(box[3])
        intervals.append((max(0, y1-margin), min(img_h, y2+margin)))
    intervals.sort(key=lambda x: x[0])
    merged = []
    curr_start, curr_end = intervals[0]
    for i in range(1, len(intervals)):
        next_start, next_end = intervals[i]
        if next_start < curr_end:
            curr_end = max(curr_end, next_end) 
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
    merged.append((curr_start, curr_end))
    return merged

# ==========================================
# 3. MAIN
# ==========================================
def main():
    frame_name = IMG_LEFT_PATH.split("/")[-1].split(_)[0]
    cprint(f"processing frame: {frame_name}","cyan")

    print(f"--- PROCESANDO FRAME: {frame_name} ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Carga Imágenes
    img_l = cv2.imread(IMG_LEFT_PATH)
    img_r = cv2.imread(IMG_RIGHT_PATH)
    h, w = img_l.shape[:2]

    # 2. Inferencia YOLO + Fish2D Creation
    print("-> YOLO Inference...")
    detector = FishDetector(MODEL_PATH) # TODO: revisar quina versió vull usar del detectortant de codi com de model.

    # Result of processin a frame
    vis_yolo, bboxes, mask_instances = detector.process_frame(img_l)
    
    # --- CREAR OBJETOS FISH2D (Para _scene.pkl) ---
    fish_list = []
    
    # Necesitamos acceder a la info detallada de YOLO para rellenar Fish2D
    # (Tu FishDetector devuelve bboxes simplificados, aquí recuperamos lógica o la añadimos)
    # Asumimos que bboxes está ordenado igual que los IDs de mask_instances (1..N)
    
    # Marco para detectar bordes (Lógica copiada de tu código antiguo)
    external_frame = np.ones((h, w), dtype=np.uint8)
    border_thickness = 10
    external_frame[border_thickness:-border_thickness, border_thickness:-border_thickness] = 0
    
    unique_ids = np.unique(mask_instances)
    fish_ids = unique_ids[unique_ids > 0]
    
    print(f"-> Creando metadatos para {len(fish_ids)} peces...")
    
    for i, fid in enumerate(fish_ids):
        # Recuperamos la info del bbox correspondiente (asumiendo orden 0 -> ID 1)
        # Esto es una aproximación, idealmente FishDetector devolvería objetos completos.
        # Si usas el FishDetector que te pasé, bboxes[i] corresponde a ID i+1.
        box = bboxes[i] # [x1, y1, x2, y2]
        
        # Check borders
        fish_mask_bool = (mask_instances == fid)
        is_touching_border = not(np.all((fish_mask_bool * external_frame) == 0))
        
        # Crear objeto Fish2D
        # OJO: Necesitas pasarle los argumentos que tu clase Fish2D espera.
        # Ajusta esto según tu definición exacta en fish2D.py
        fish_obj = Fish2D(
            frame_id=frame_name,
            color_id=fid, # Usamos el ID como color simbólico
            model_classes_dict=detector.fish_dict,
            class_colours_dict=detector.class_colours,
            fish_class=0, # TODO: Recuperar la clase real de YOLO si FishDetector la guarda
            track_id=fid,
            in_image_borders=is_touching_border,
            debug_path=os.path.join(OUTPUT_DIR, "debug")
        )
        # Hack: Asignamos manualmente propiedades si el constructor no las pide
        # fish_obj.bbox = box 
        fish_list.append(fish_obj)

    # 3. Stereo Pipeline (Recorte + SGBM + WLS)
    print("-> Stereo Pipeline...")
    
    # ... (Decisión de Franjas y Preproceso igual que antes) ...
    candidate_strips = merge_strips(bboxes, h, margin=NUM_DISP + 20)
    total_strip_height = sum([y2 - y1 for (y1, y2) in candidate_strips])
    coverage_ratio = total_strip_height / float(h)
    
    if len(bboxes) > 0 and coverage_ratio < 0.8:
        strips = candidate_strips
    else:
        strips = [(0, h)]

    gray_l = (0.7 * img_l[:,:,1] + 0.3 * img_l[:,:,0]).astype(np.uint8)
    gray_r = (0.7 * img_r[:,:,1] + 0.3 * img_r[:,:,0]).astype(np.uint8)
    gray_r = match_histogram_stats(gray_r, gray_l)
    
    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_GRID, CLAHE_GRID))
        gray_l = clahe.apply(gray_l)
        gray_r = clahe.apply(gray_r)

    left_matcher = cv2.StereoSGBM_create(minDisparity=MIN_DISP, numDisparities=NUM_DISP, blockSize=BLOCK_SIZE,
                                         P1=8*3*BLOCK_SIZE**2, P2=32*3*BLOCK_SIZE**2, disp12MaxDiff=1,
                                         uniquenessRatio=UNIQUENESS, speckleWindowSize=SPECKLE_WIN, speckleRange=SPECKLE_RNG,
                                         mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    wls_filter.setLambda(WLS_LAMBDA); wls_filter.setSigmaColor(WLS_SIGMA)

    full_disp = np.zeros((h, w), dtype=np.int16)
    
    for (y1, y2) in strips:
        crop_l = gray_l[y1:y2, :]
        crop_r = gray_r[y1:y2, :]
        disp_l = left_matcher.compute(crop_l, crop_r)
        disp_r = right_matcher.compute(crop_r, crop_l)
        filtered = wls_filter.filter(disp_l, img_l[y1:y2, :], disparity_map_right=disp_r)
        full_disp[y1:y2, :] = filtered

    # 4. Generar Nube de Puntos Segmentada (Formato Antiguo)
    print("-> Generando datos compatibles con sistema antiguo...")
    
    disp_float = full_disp.astype(np.float32) / 16.0
    Q = get_decimated_Q()
    points_3d = cv2.reprojectImageTo3D(disp_float, Q)
    
    # Seleccionamos puntos válidos: que tengan profundidad y sean pez
    # (Tu código antiguo espera que los puntos tengan ID > 0)
    valid_mask = (disp_float > min_valid_disparity) & (mask_instances > 0)
    
    if np.count_nonzero(valid_mask) > 0:
        # Extraemos XYZ
        pts_xyz = points_3d[valid_mask]
        # Extraemos IDs correspondientes
        pts_ids = mask_instances[valid_mask].reshape(-1, 1) # Forma (N, 1)
        
        # Concatenamos: [X, Y, Z, ID]
        # Tu código antiguo espera este formato en el PCD
        segmented_pc_data = np.hstack((pts_xyz, pts_ids)).astype(np.float32)
        
        # --- GUARDAR 1: _segmented.pcd ---
        # Como Open3D no guarda 4 canales fácilmente en .pcd sin trucos, 
        # y tu 'utils.read_fish_scene_pc' seguramente usa open3d o numpy,
        # lo más robusto aquí es guardar un .ply con campo escalar extra o un .xyz
        
        # Opción A: Guardar como .ply usando Open3D pero "hackeando" el color para guardar el ID
        # (Si tu código antiguo lee el color como ID, usa esto. Si lee una 4a columna, usa Opción B)
        
        # Opción B (Recomendada para compatibilidad científica): Guardar como ASCII XYZI o binario custom.
        # Pero dado que open3d es estricto, vamos a guardar como .pcd usando la librería pypcd o cabeceras manuales.
        # O MEJOR AÚN: Guardémoslo como un archivo PointCloud de Open3D estándar donde
        # X,Y,Z son los puntos y usamos colores falsos para el ID, 
        # PERO tu código usa `pointcloud_info[:,-1]`, así que espera una matriz NumPy.
        
        # TRUCO: Guardar como .pcd ASCII manualmente es fácil y funciona siempre.
        output_pcd_path = os.path.join(OUTPUT_DIR, f"{frame_name}_segmented.pcd")
        
        # Creamos una nube open3d normal
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_xyz)
        # Guardamos el ID en el canal de Intensidad (si el lector lo soporta) o creamos tensores.
        # Dado que no se cómo es tu 'utils.read_fish_scene_pc', vamos a asumir que
        # puede leer un PCD estándar.
        
        # Si tu utils usa `pcd = o3d.io.read_point_cloud`, Open3D NO lee 4 columnas por defecto como ID.
        # Si usa `pyntcloud` o `pandas`, sí.
        
        # PROBEMOS ESTO: Guardar con numpy (para depuración) y PCD.
        # Para que tu código funcione SIN CAMBIOS, voy a guardar un PCD donde 'rgb' codifique el ID,
        # o asumiré que modificas ligeramente 'utils.py'.
        
        # SOLUCIÓN MÁS SEGURA: Guardar los puntos. El código de medida ya lo lee.
        # Vamos a guardar usando Open3D pero meteremos los IDs en 'colors' (normalizado) para visualización
        # y guardaremos un archivo .npy auxiliar si hace falta.
        
        # Pero espera, tu código dice: `pointcloud_info[:,-1]==object_id`.
        # Esto implica que carga una matriz Nx4.
        
        # Hack para Open3D para guardar IDs (guardamos como float en un canal custom no es trivial).
        # Vamos a escribir el PCD a mano, es lo más seguro para formato Nx4.
        with open(output_pcd_path, 'w') as f:
            f.write("# .PCD v0.7 - Data 3D - Fish Segmented\n")
            f.write("VERSION 0.7\nFIELDS x y z label\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n")
            f.write(f"WIDTH {len(segmented_pc_data)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(segmented_pc_data)}\nDATA ascii\n")
            for row in segmented_pc_data:
                f.write(f"{row[0]:.4f} {row[1]:.4f} {row[2]:.4f} {int(row[3])}\n")
        
        print(f"✅ Guardado PCD segmentado: {output_pcd_path}")

        # --- GUARDAR 2: _scene.pkl ---
        scene_pkl_path = os.path.join(OUTPUT_DIR, f"{frame_name}_scene.pkl")
        
        # Creamos el objeto FrameScene
        frame_scene = FrameScene(
            frame_name=frame_name,
            fish_list=fish_list,
            object_ids_mask=mask_instances, # Guardamos la máscara completa
            class_ids_img=mask_instances,   # Simplificación
            disparity_img=full_disp,        # Guardamos disparidad cruda para debug
            save_path=OUTPUT_DIR
        )
        
        with open(scene_pkl_path, "wb") as f:
            pickle.dump(frame_scene, f)
            
        print(f"✅ Guardado Scene Pickle: {scene_pkl_path}")
        
    else:
        print("⚠️ No hay puntos válidos para guardar.")

    print("-> ¡PROCESO TERMINADO!")

if __name__ == '__main__':
    main()