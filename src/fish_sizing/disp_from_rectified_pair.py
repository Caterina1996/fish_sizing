import numpy as np
import cv2
import sys
import os

# ==========================================
#              CONFIGURACIÓN
# ==========================================

IMG_LEFT_PATH = '/home/rosuser/repo/src/out/LIMA/2025/2025_08_21/selec2/12_07_31/frame236_left_image.jpg'   
IMG_RIGHT_PATH = '/home/rosuser/repo/src/out/LIMA/2025/2025_08_21/selec2/12_07_31/frame236_right_image.jpg' 

TARGET_DISPLAY_WIDTH = 1600 

# ==========================================
def nothing(x):
    pass

def main():
    # 1. Cargar imágenes
    if not os.path.exists(IMG_LEFT_PATH) or not os.path.exists(IMG_RIGHT_PATH):
        print(f"Error: No encuentro las imágenes.")
        sys.exit(1)

    img_l = cv2.imread(IMG_LEFT_PATH)
    img_r = cv2.imread(IMG_RIGHT_PATH)
    
    print(f"Imágenes cargadas. Resolución: {img_l.shape}")

    # 2. Pre-proceso BASE (Mezcla de Canales Verde/Azul)
    # Guardamos esto como "base" para no sobreescribirlo en el bucle
    gray_l_base = (0.7 * img_l[:,:,1] + 0.3 * img_l[:,:,0]).astype(np.uint8)
    gray_r_base = (0.7 * img_r[:,:,1] + 0.3 * img_r[:,:,0]).astype(np.uint8)

    # 3. Configurar Ventana
    window_name = "Stereo Tuner (CLAHE + SGBM)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 

    # 4. Crear Trackbars
    # --- NUEVOS CONTROLES PARA CLAHE ---
    cv2.createTrackbar('USE CLAHE', window_name, 0, 1, nothing)        # 0=Off, 1=On
    cv2.createTrackbar('CLAHE Clip (x10)', window_name, 30, 80, nothing) # Default 3.0
    cv2.createTrackbar('CLAHE Grid', window_name, 8, 32, nothing)      # Default 8x8
    
    # --- CONTROLES SGBM ---
    cv2.createTrackbar('Num Disps (x16)', window_name, 6, 20, nothing)
    cv2.createTrackbar('Min Disp', window_name, 0, 100, nothing)
    cv2.createTrackbar('Block Size', window_name, 5, 25, nothing)
    cv2.createTrackbar('Uniqueness', window_name, 10, 50, nothing)
    cv2.createTrackbar('Speckle Win', window_name, 100, 200, nothing)
    cv2.createTrackbar('Speckle Rng', window_name, 32, 50, nothing)
    
    # --- CONTROLES WLS ---
    cv2.createTrackbar('WLS Lambda', window_name, 15, 80, nothing) 
    cv2.createTrackbar('WLS Sigma', window_name, 10, 30, nothing)    

    print("\n--- INSTRUCCIONES ---")
    print("Usa el slider 'USE CLAHE' para activar la corrección de iluminación.")
    print("Pulsa 'q' para guardar los valores y salir.")

    while True:
        # Leer Sliders de CLAHE
        use_clahe = cv2.getTrackbarPos('USE CLAHE', window_name)
        clahe_clip = cv2.getTrackbarPos('CLAHE Clip (x10)', window_name) / 10.0
        clahe_grid = cv2.getTrackbarPos('CLAHE Grid', window_name)
        if clahe_grid < 1: clahe_grid = 1 # Evitar error por 0

        # Leer Sliders SGBM
        n_disp_mult = cv2.getTrackbarPos('Num Disps (x16)', window_name)
        min_disp = cv2.getTrackbarPos('Min Disp', window_name)
        block_size = cv2.getTrackbarPos('Block Size', window_name)
        uniqueness = cv2.getTrackbarPos('Uniqueness', window_name)
        speckle_win = cv2.getTrackbarPos('Speckle Win', window_name)
        speckle_rng = cv2.getTrackbarPos('Speckle Rng', window_name)
        wls_lambda = cv2.getTrackbarPos('WLS Lambda', window_name) * 100.0
        wls_sigma = cv2.getTrackbarPos('WLS Sigma', window_name) / 10.0

        # Validaciones
        if n_disp_mult < 1: n_disp_mult = 1
        num_disp = n_disp_mult * 16
        if block_size % 2 == 0: block_size += 1
        if block_size < 3: block_size = 3

        # --- APLICAR CLAHE (Si está activo) ---
        # Trabajamos sobre copias para no alterar la base
        gray_l = gray_l_base.copy()
        gray_r = gray_r_base.copy()

        if use_clahe == 1:
            # Crear objeto CLAHE con los parámetros dinámicos
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
            gray_l = clahe.apply(gray_l)
            gray_r = clahe.apply(gray_r)

        # Matchers
        left_matcher = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=uniqueness,
            speckleWindowSize=speckle_win,
            speckleRange=speckle_rng,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        
        # Calcular Disparidad
        disp_l = left_matcher.compute(gray_l, gray_r)
        disp_r = right_matcher.compute(gray_r, gray_l)

        # Filtro WLS
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(wls_lambda)
        wls_filter.setSigmaColor(wls_sigma)
        disp_filtered = wls_filter.filter(disp_l, img_l, disparity_map_right=disp_r)

        # Visualización
        vis = (disp_filtered.astype(np.float32) / 16.0 - min_disp) / num_disp
        vis = np.clip(vis, 0, 1)
        vis = (vis * 255).astype(np.uint8)
        vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)

        # Máscara de error
        invalid = (disp_filtered < (min_disp * 16))
        vis_color[invalid] = 0

        # --- VISUALIZACIÓN COMPARATIVA ---
        # Queremos ver: Imagen Original Izq | Imagen Procesada Izq (CLAHE) | Disparidad
        
        # Convertimos la procesada a BGR para poder apilarla
        processed_view = cv2.cvtColor(gray_l, cv2.COLOR_GRAY2BGR)
        
        # Apilamos 3 cosas: Original, Procesada (para ver el efecto de CLAHE), Disparidad
        combined = np.hstack((img_l, processed_view, vis_color))

        # Reescalado
        h, w = combined.shape[:2]
        scale_factor = TARGET_DISPLAY_WIDTH / float(w)
        new_dim = (TARGET_DISPLAY_WIDTH, int(h * scale_factor))
        final_view = cv2.resize(combined, new_dim, interpolation=cv2.INTER_AREA)
        
        # Texto informativo sobre la imagen
        cv2.putText(final_view, "Original", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(final_view, "Procesado (CLAHE)", (int(TARGET_DISPLAY_WIDTH/3) + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(final_view, "Disparidad", (int(2*TARGET_DISPLAY_WIDTH/3) + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow(window_name, final_view)

        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            print("\n" + "="*30)
            print(">>> PARÁMETROS FINALES <<<")
            if use_clahe:
                print(f"CLAHE_ENABLED = True")
                print(f"CLAHE_CLIP = {clahe_clip}")
                print(f"CLAHE_GRID = ({clahe_grid}, {clahe_grid})")
            else:
                print("CLAHE_ENABLED = False")
            print(f"MIN_DISP = {min_disp}")
            print(f"NUM_DISP = {num_disp}")
            print(f"BLOCK_SIZE = {block_size}")
            print(f"UNIQUENESS = {uniqueness}")
            print("="*30)
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()