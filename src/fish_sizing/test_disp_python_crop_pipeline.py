import cv2
import numpy as np
import open3d as o3d
import sys
import os

# Importar el detector
try:
    from fish_detector import FishDetector
except ImportError as e:
    print(f"Error importando fish_detector: {e}")
    sys.exit(1)

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
IMG_LEFT_PATH = '/home/rosuser/repo/src/out/LIMA/test_cat/frame625_left_image.jpg'
IMG_RIGHT_PATH = '/home/rosuser/repo/src/out/LIMA/test_cat/frame625_right_image.jpg'
MODEL_PATH = "/home/rosuser/repo/dataset/models/yv11l/ylarge_dfishthings_d18_poolv2r_lantytr_nocturnes/weights/best.pt"
OUTPUT_DIR = "/home/rosuser/repo/src/out/LIMA/test_cat/"

# --- OPCIONES DE GUARDADO ---
SAVE_INDIVIDUAL_FISH = True   # Guardar un .ply por cada pez (fish_1.ply, fish_2.ply...)
SAVE_COMBINED_SCENE = True    # Guardar todos los peces juntos (scene_combined.ply)
SAVE_RAW_STRIP = False        # Guardar TODO lo que tenga disparidad (incluido fondo/paredes) en la franja

# Params Pre-proceso
USE_CLAHE = True
CLAHE_CLIP = 3.0 
CLAHE_GRID = 8 

# STEREO SGBM
MIN_DISP = 0        
NUM_DISP = 96
BLOCK_SIZE = 5
UNIQUENESS = 10
SPECKLE_WIN = 150
SPECKLE_RNG = 32

# WLS Filter
WLS_LAMBDA = 8000.0
WLS_SIGMA = 1.0 # Bajo para preservar bordes

# Calibración (Decimada x2)
FOCAL_DECIMATED = 730.35 
BASELINE = 0.1203 
CX = 520.70  
CY = 380.34 

# Filtro de Profundidad Física
MAX_DEPTH_METERS = 4.0
min_valid_disparity = (FOCAL_DECIMATED * BASELINE) / MAX_DEPTH_METERS

# ==========================================
# 2. FUNCIONES
# ==========================================
def get_decimated_Q():
    return np.float32([
        [1, 0, 0, -CX],
        [0, 1, 0, -CY],
        [0, 0, 0, FOCAL_DECIMATED],
        [0, 0, 1.0/BASELINE, 0]  # Positivo para Z hacia adelante
    ])

def match_histogram_stats(source, reference):
    m_src, s_src = cv2.meanStdDev(source)
    m_ref, s_ref = cv2.meanStdDev(reference)
    if s_src[0][0] < 1e-3: return source
    ratio = s_ref[0][0] / s_src[0][0]
    res = (source.astype(np.float32) - m_src[0][0]) * ratio + m_ref[0][0]
    return np.clip(res, 0, 255).astype(np.uint8)

def merge_strips(bboxes, img_h, margin=50):
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

def save_point_cloud(points_3d, colors, mask, filename, z_min=0.1, z_max=10.0):
    """
    Función auxiliar para filtrar y guardar nubes de puntos.
    points_3d: Imagen (H,W,3) con coordenadas XYZ
    colors: Imagen (H,W,3) con colores BGR
    mask: Máscara booleana de los píxeles que queremos guardar
    """
    # 1. Aplicar máscara de selección (ej. solo pez 1)
    valid_points = points_3d[mask]
    valid_colors = colors[mask]
    
    if len(valid_points) == 0:
        print(f"   [WARN] No hay puntos para {filename} (Mask vacía).")
        return

    # 2. Filtrar por profundidad (Z) para quitar ruido lejano
    zs = valid_points[:, 2]
    z_mask = (zs > z_min) & (zs < z_max)
    
    final_points = valid_points[z_mask]
    final_colors = valid_colors[z_mask]
    
    if len(final_points) > 0:
        print(f"   -> Guardando {filename} ({len(final_points)} puntos)...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(final_points)
        # Convertir BGR a RGB para Open3D
        pcd.colors = o3d.utility.Vector3dVector(final_colors[:, ::-1] / 255.0)
        
        full_path = os.path.join(OUTPUT_DIR, filename)
        o3d.io.write_point_cloud(full_path, pcd)
    else:
        print(f"   [WARN] {filename} descartado (0 puntos tras filtro Z).")

# ==========================================
# 3. MAIN
# ==========================================
def main():
    print("--- INICIANDO PIPELINE MULTI-OBJETO ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Carga
    img_l = cv2.imread(IMG_LEFT_PATH)
    img_r = cv2.imread(IMG_RIGHT_PATH)
    if img_l is None: sys.exit("Error cargando imagenes")
    h, w = img_l.shape[:2]

    # 2. Inferencia YOLO
    print("-> Ejecutando YOLO...")
    detector = FishDetector(MODEL_PATH)
    vis_yolo, bboxes, mask_instances = detector.process_frame(img_l)
    print(f"-> Detectados {len(bboxes)} peces.")

    # 3. Franjas
    candidate_strips = merge_strips(bboxes, h, margin=NUM_DISP + 20)
    total_strip_height = sum([y2 - y1 for (y1, y2) in candidate_strips])
    coverage_ratio = total_strip_height / float(h)
    
    FORCE_FULL_IMAGE_THRESHOLD = 0.80 

    if len(bboxes) > 0 and coverage_ratio < FORCE_FULL_IMAGE_THRESHOLD:
        strips = candidate_strips
        print(f"   -> Modo FRANJAS: {strips} (Ocupación: {coverage_ratio:.1%})")
    else:
        strips = [(0, h)]
        print("   -> Modo IMAGEN COMPLETA.")

    # 4. Pre-Proceso
    gray_l = (0.7 * img_l[:,:,1] + 0.3 * img_l[:,:,0]).astype(np.uint8)
    gray_r = (0.7 * img_r[:,:,1] + 0.3 * img_r[:,:,0]).astype(np.uint8)
    gray_r = match_histogram_stats(gray_r, gray_l)
    
    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_GRID, CLAHE_GRID))
        gray_l = clahe.apply(gray_l)
        gray_r = clahe.apply(gray_r)

    # 5. Stereo SGBM + WLS
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=MIN_DISP, numDisparities=NUM_DISP, blockSize=BLOCK_SIZE,
        P1=8*3*BLOCK_SIZE**2, P2=32*3*BLOCK_SIZE**2, disp12MaxDiff=1,
        uniquenessRatio=UNIQUENESS, speckleWindowSize=SPECKLE_WIN, speckleRange=SPECKLE_RNG,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    wls_filter.setLambda(WLS_LAMBDA); wls_filter.setSigmaColor(WLS_SIGMA)

    print("-> Calculando Disparidad...")
    full_disp = np.zeros((h, w), dtype=np.int16)
    debug_crop_img = img_l.copy()

    for (y1, y2) in strips:
        crop_l = gray_l[y1:y2, :]
        crop_r = gray_r[y1:y2, :]
        disp_l = left_matcher.compute(crop_l, crop_r)
        disp_r = right_matcher.compute(crop_r, crop_l)
        filtered = wls_filter.filter(disp_l, img_l[y1:y2, :], disparity_map_right=disp_r)
        full_disp[y1:y2, :] = filtered
        cv2.rectangle(debug_crop_img, (0, y1), (w, y2), (0, 255, 0), 2)

    # 6. Reproyección 3D
    print("-> Reproyectando a 3D...")
    disp_float = full_disp.astype(np.float32) / 16.0
    Q = get_decimated_Q()
    points_3d = cv2.reprojectImageTo3D(disp_float, Q)

    # 7. GUARDADO DE NUBES DE PUNTOS
    
    # Base Mask: Píxeles con disparidad válida (y cercanos)
    valid_disp_mask = (disp_float > min_valid_disparity)

    # A) Nube de todos los peces juntos (Limpia)
    if SAVE_COMBINED_SCENE:
        # Puntos que tienen disp válida Y pertenecen a ALGÚN pez (>0)
        combined_mask = valid_disp_mask & (mask_instances > 0)
        save_point_cloud(points_3d, img_l, combined_mask, "scene_combined.ply", z_max=MAX_DEPTH_METERS)

    # B) Nubes individuales por pez
    if SAVE_INDIVIDUAL_FISH:
        # Encontrar qué IDs existen en la imagen (quitando el 0 que es fondo)
        unique_ids = np.unique(mask_instances)
        fish_ids = unique_ids[unique_ids > 0] 
        
        print(f"-> Separando {len(fish_ids)} peces individuales...")
        for fid in fish_ids:
            # Máscara específica para este pez
            fish_mask = valid_disp_mask & (mask_instances == fid)
            save_point_cloud(points_3d, img_l, fish_mask, f"fish_ID_{fid}.ply", z_max=MAX_DEPTH_METERS)

    # C) (Opcional) Nube "Raw" de la franja (incluyendo fondo/ruido)
    if SAVE_RAW_STRIP:
        # Solo miramos disparidad, ignoramos lo que diga YOLO
        # Útil para ver si hemos cortado la cola del pez por culpa de la máscara
        save_point_cloud(points_3d, img_l, valid_disp_mask, "strip_raw_background.ply", z_max=MAX_DEPTH_METERS)

    # Guardar imágenes de debug
    cv2.imwrite(os.path.join(OUTPUT_DIR,"out_1_yolo.jpg"), vis_yolo)
    cv2.imwrite(os.path.join(OUTPUT_DIR,"out_3_disparity.jpg"), cv2.applyColorMap(((disp_float-MIN_DISP)/NUM_DISP*255).astype(np.uint8), cv2.COLORMAP_JET))
    print("-> ¡PROCESO TERMINADO!")

if __name__ == '__main__':
    main()