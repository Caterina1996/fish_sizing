import cv2
import numpy as np
import argparse
import sys
from termcolor import cprint

# Importar tu propia clase de procesado
from fish_sizing.img_processing.image_processor import ImageProcessor

# --- CONFIGURACIÓN ---
PATH_MAPPINGS = {
    "/home/slimbook/bagfiles": "/home/rosuser/dataset/bagfiles",
    "/home/slimbook/fish_sizing/out": "/home/rosuser/repo/out",
    "/home/slimbook/models": "/home/rosuser/dataset/models/",
    "/home/slimbook/fish_sizing/config" :"/home/rosuser/repo/config/",
    "/media/slimbook/easystore": "/home/rosuser/easystore"
}

USE_DOCKER = True

def nothing(x):
    pass

def transform_path2docker(path: str) -> str:
    if not USE_DOCKER or path is None:
        return path
    for host_path, docker_path in PATH_MAPPINGS.items():
        if host_path in path:
            new_path = path.replace(host_path, docker_path)
            cprint(f"🔄 Path mapped: {path} \n   -> {new_path}", "yellow")
            return new_path 
    return path

def main():
    parser = argparse.ArgumentParser(description="Afinador interactivo de SGBM + WLS + Image Processing")
    
    parser.add_argument("--left",  help="Ruta a la imagen izquierda (rectificada)", 
    default="/media/slimbook/easystore/results_fish_sizing/seleccio_article/2024_11_12/test_stereo_config/2024_11_12-11_37-frame_38_left.png")
    
    parser.add_argument("--right", help="Ruta a la imagen derecha (rectificada)",
    default="/media/slimbook/easystore/results_fish_sizing/seleccio_article/2024_11_12/test_stereo_config/2024_11_12-11_37-frame_38_right.png")
    
    parser.add_argument("--scale", type=float, default=0.5, help="Escala para que vaya fluido (def: 0.5)")
    args = parser.parse_args()

    imgL_raw = cv2.imread(transform_path2docker(args.left))
    imgR_raw = cv2.imread(transform_path2docker(args.right))

    if imgL_raw is None or imgR_raw is None:
        print("❌ Error cargando imágenes. Revisa las rutas.")
        sys.exit(1)

    if args.scale != 1.0:
        imgL_raw = cv2.resize(imgL_raw, (0,0), fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        imgR_raw = cv2.resize(imgR_raw, (0,0), fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)

    processor = ImageProcessor(stereo=True)
    processor.set_image_pair(imgL_raw, imgR_raw)
    
    pipelines = {
        0: ("Raw (Sin Filtros)", []),
        1: ("Basic (Gris + CLAHE)", [
            ("convert_to_custom_grayscale", {}, False),
            ("apply_clahe", {"clip_limit": 2.0, "grid_size": (8,8)}, False)
        ]),
        2: ("Sharp (Gamma + Sharp)", [
            ("apply_gamma", {"gamma": 1.2}, False),
            ("apply_sharpen", {"alpha": 1.7}, False),
            ("convert_to_custom_grayscale", {}, False)
        ]),
        3: ("Clean Edges (Bilateral + CLAHE)", [
            ("apply_bilateral", {"d": 7, "sigma_color": 50, "sigma_space": 50}, False),
            ("convert_to_custom_grayscale", {}, False),
            ("apply_clahe", {"clip_limit": 1.5, "grid_size": (8,8)}, False)
        ])
    }

    cv2.namedWindow('Stereo Tuning', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Stereo Tuning', 1400, 900)

    # --- TRACKBARS ---
    cv2.createTrackbar('Pipeline', 'Stereo Tuning', 0, 3, nothing)
    
    # Parámetros Base Estéreo
    cv2.createTrackbar('minDisparity', 'Stereo Tuning', 1, 30, nothing)
    cv2.createTrackbar('numDisparities (x16)', 'Stereo Tuning', 2, 15, nothing)
    cv2.createTrackbar('blockSize', 'Stereo Tuning', 3, 15, nothing) 
    cv2.createTrackbar('Mode (0=SGBM,1=HH,2=3WAY,3=HH4)', 'Stereo Tuning', 2, 3, nothing)
    
    # Factores P1, P2 y validación
    cv2.createTrackbar('P1_mult', 'Stereo Tuning', 8, 32, nothing)  
    cv2.createTrackbar('P2_mult', 'Stereo Tuning', 24, 80, nothing) 
    cv2.createTrackbar('disp12MaxDiff', 'Stereo Tuning', 1, 20, nothing)
    
    # Post-filtros clásicos
    cv2.createTrackbar('uniquenessRatio', 'Stereo Tuning', 25, 50, nothing)
    cv2.createTrackbar('speckleWinSize', 'Stereo Tuning', 50, 300, nothing)
    cv2.createTrackbar('speckleRange', 'Stereo Tuning', 2, 32, nothing)
    
    # Filtro WLS
    cv2.createTrackbar('Use WLS (0=No, 1=Si)', 'Stereo Tuning', 1, 1, nothing)
    cv2.createTrackbar('WLS Lambda (x100)', 'Stereo Tuning', 20, 100, nothing) # 20 = 2000
    cv2.createTrackbar('WLS Sigma (x0.1)', 'Stereo Tuning', 8, 30, nothing)    # 8 = 0.8

    last_pipeline_idx = -1
    proc_L, proc_R = None, None

    # Mapeo de modos SGBM
    mode_map = {
        0: cv2.STEREO_SGBM_MODE_SGBM,
        1: cv2.STEREO_SGBM_MODE_HH,
        2: cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        3: cv2.STEREO_SGBM_MODE_HH4
    }
    mode_names = ["MODE_SGBM", "MODE_HH", "MODE_SGBM_3WAY", "MODE_HH4"]

    print("\n🚀 INICIANDO ENTORNO DE PRUEBAS AVANZADO")
    print("Mueve los deslizadores. Presiona ESC para salir e imprimir los mejores valores.\n")

    while True:
        # 1. Leer parámetros SGBM
        minDisp = cv2.getTrackbarPos('minDisparity', 'Stereo Tuning')
        numDisp = max(1, cv2.getTrackbarPos('numDisparities (x16)', 'Stereo Tuning')) * 16
        bSize = cv2.getTrackbarPos('blockSize', 'Stereo Tuning') * 2 + 1 
        mode_idx = cv2.getTrackbarPos('Mode (0=SGBM,1=HH,2=3WAY,3=HH4)', 'Stereo Tuning')
        
        p1_m = max(1, cv2.getTrackbarPos('P1_mult', 'Stereo Tuning'))
        p2_m = max(p1_m + 1, cv2.getTrackbarPos('P2_mult', 'Stereo Tuning'))
        d12Max = cv2.getTrackbarPos('disp12MaxDiff', 'Stereo Tuning')
        
        uRatio = cv2.getTrackbarPos('uniquenessRatio', 'Stereo Tuning')
        spkW = cv2.getTrackbarPos('speckleWinSize', 'Stereo Tuning')
        spkR = cv2.getTrackbarPos('speckleRange', 'Stereo Tuning')
        
        # Parámetros WLS
        use_wls = cv2.getTrackbarPos('Use WLS (0=No, 1=Si)', 'Stereo Tuning') == 1
        wls_lam = cv2.getTrackbarPos('WLS Lambda (x100)', 'Stereo Tuning') * 100.0
        wls_sig = cv2.getTrackbarPos('WLS Sigma (x0.1)', 'Stereo Tuning') / 10.0

        # 2. Pipeline de Imagen
        curr_pipeline_idx = cv2.getTrackbarPos('Pipeline', 'Stereo Tuning')
        
        if curr_pipeline_idx != last_pipeline_idx:
            pipe_name, steps = pipelines[curr_pipeline_idx]
            print(f"🔄 Cambiando a Pipeline: {pipe_name}")
            
            processor.reset()
            processor.run_pipeline(steps, visualize=False)
            proc_L, proc_R = processor.get_processed()
            
            if len(proc_L.shape) == 3:
                proc_L_gray = cv2.cvtColor(proc_L, cv2.COLOR_BGR2GRAY)
                proc_R_gray = cv2.cvtColor(proc_R, cv2.COLOR_BGR2GRAY)
            else:
                proc_L_gray, proc_R_gray = proc_L, proc_R
                
            last_pipeline_idx = curr_pipeline_idx

        # 3. Configurar Estéreo
        stereo = cv2.StereoSGBM_create(
            minDisparity=minDisp,
            numDisparities=numDisp,
            blockSize=bSize,
            P1=p1_m * 1 * bSize**2,
            P2=p2_m * 1 * bSize**2,
            disp12MaxDiff=d12Max,
            uniquenessRatio=uRatio,
            speckleWindowSize=spkW,
            speckleRange=spkR,
            mode=mode_map[mode_idx]
        )

        # 4. Calcular Disparidad (Con o sin WLS)
        if use_wls:
            # Para WLS necesitamos calcular la disparidad de ambos lados
            right_matcher = cv2.ximgproc.createRightMatcher(stereo)
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
            wls_filter.setLambda(wls_lam)
            wls_filter.setSigmaColor(wls_sig)
            
            dispL = stereo.compute(proc_L_gray, proc_R_gray)
            dispR = right_matcher.compute(proc_R_gray, proc_L_gray)
            
            # El filtro WLS usa la imagen original para respetar los bordes
            filtered_disp = wls_filter.filter(dispL, proc_L, None, dispR)
            disparity = filtered_disp.astype(np.float32) / 16.0
        else:
            disparity = stereo.compute(proc_L_gray, proc_R_gray).astype(np.float32) / 16.0
        
        # 5. Visualización
        # Cortar a minDisp para evitar que el negro del margen estropee el mapa de color
        disparity_viz = np.clip(disparity, minDisp, numDisp)
        disp_viz_norm = cv2.normalize(disparity_viz, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disp_color = cv2.applyColorMap(np.uint8(disp_viz_norm), cv2.COLORMAP_JET)

        if len(proc_L.shape) == 2:
            show_L = cv2.cvtColor(proc_L, cv2.COLOR_GRAY2BGR)
        else:
            show_L = proc_L.copy()

        cv2.putText(show_L, f"Pipeline: {pipelines[curr_pipeline_idx][0]}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        wls_text = f"WLS: ON (L={int(wls_lam)}, S={wls_sig})" if use_wls else "WLS: OFF"
        cv2.putText(disp_color, wls_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        combined = np.hstack((show_L, disp_color))
        cv2.imshow('Stereo Tuning', combined)
        
        if cv2.waitKey(30) == 27: # ESC
            break
            
    cv2.destroyAllWindows()
    
    print("\n" + "="*40)
    print(" 💾 TU NUEVO STEREO_CONFIG.YAML 💾 ")
    print("="*40)
    print(f"image_processing_pipeline: {pipelines[curr_pipeline_idx][0]}")
    print("stereo:")
    print(f"  min_disparity:       {minDisp}")
    print(f"  num_disparities:     {numDisp}")
    print(f"  block_size:          {bSize}")
    print(f"  p1_factor:           {p1_m}")
    print(f"  p2_factor:           {p2_m}")
    print(f"  disp12_max_diff:     {d12Max}")
    print(f"  uniqueness_ratio:    {uRatio}")
    print(f"  speckle_window_size: {spkW}")
    print(f"  speckle_range:       {spkR}")
    print(f"  mode:                \"{mode_names[mode_idx]}\"")
    print("wls:")
    print(f"  use_wls:             {use_wls}")
    if use_wls:
        print(f"  lambda:              {wls_lam}")
        print(f"  sigma:               {wls_sig}")
    print("="*40 + "\n")

if __name__ == '__main__':
    main()