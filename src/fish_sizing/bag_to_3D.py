import rosbag
import numpy as np
import cv2
from cv_bridge import CvBridge
import os

# --- CONFIGURACIÓN ---
BAGFILE = '/home/rosuser/repo/dataset/bagfiles/LIMA/2025/2025_08_21/10_36_30/stereo_camera_images_2025-08-21-10-36-31_0.bag' 

OUT_DIR = '/home/rosuser/repo/src/out/LIMA/2025/2025_08_21/test_python/'

LEFT_TOPIC = '/stereo_ch3/left/image_raw'
RIGHT_TOPIC = '/stereo_ch3/right/image_raw'
LEFT_INFO = '/stereo_ch3/left/camera_info'
RIGHT_INFO = '/stereo_ch3/right/camera_info'

# Parámetros Stereo (Ajustados para x2)
MIN_DISP = 0
NUM_DISP = 128      # Rango de búsqueda (debe ser divisible por 16)
BLOCK_SIZE = 5      # Tamaño de ventana (3-7 suele ir bien para peces)


def get_calibration_and_one_frame(bag_path):
    bag = rosbag.Bag(bag_path)
    bridge = CvBridge()
    
    info_l, info_r = None, None
    img_l, img_r = None, None
    
    # Buscamos calibración e imágenes
    # Leemos un poco más de mensajes por si no están sincronizados al principio
    for topic, msg, t in bag.read_messages(topics=[LEFT_INFO, RIGHT_INFO, LEFT_TOPIC, RIGHT_TOPIC]):
        if topic == LEFT_INFO and info_l is None: info_l = msg
        elif topic == RIGHT_INFO and info_r is None: info_r = msg
        elif topic == LEFT_TOPIC and img_l is None: img_l = msg
        elif topic == RIGHT_TOPIC and img_r is None: img_r = msg
        
        if all(x is not None for x in [info_l, info_r, img_l, img_r]):
            break
    
    bag.close()
    
    if img_l is None or img_r is None:
        raise ValueError("No se encontraron imágenes en el bag")

    # Procesar Calibración
    K_l, D_l = np.array(info_l.K).reshape(3,3), np.array(info_l.D)
    R_l, P_l = np.array(info_l.R).reshape(3,3), np.array(info_l.P).reshape(3,4)
    K_r, D_r = np.array(info_r.K).reshape(3,3), np.array(info_r.D)
    R_r, P_r = np.array(info_r.R).reshape(3,3), np.array(info_r.P).reshape(3,4)
    size = (info_l.width, info_l.height)
    
    m_l1, m_l2 = cv2.initUndistortRectifyMap(K_l, D_l, R_l, P_l, size, cv2.CV_16SC2)
    m_r1, m_r2 = cv2.initUndistortRectifyMap(K_r, D_r, R_r, P_r, size, cv2.CV_16SC2)
    
    # Procesar Imágenes
    cv_l = bridge.imgmsg_to_cv2(img_l, "bgr8")
    cv_r = bridge.imgmsg_to_cv2(img_r, "bgr8")
    
    # Rectificar
    rect_l = cv2.remap(cv_l, m_l1, m_l2, cv2.INTER_LINEAR)
    rect_r = cv2.remap(cv_r, m_r1, m_r2, cv2.INTER_LINEAR)
    
    # Decimate x2 (Opcional, pero recomendado)
    rect_l = cv2.resize(rect_l, (0,0), fx=0.5, fy=0.5)
    rect_r = cv2.resize(rect_r, (0,0), fx=0.5, fy=0.5)
    
    return rect_l, rect_r

def nothing(x):
    pass

def tuner():
    print("Cargando imágenes...")
    try:
        img_l, img_r = get_calibration_and_one_frame(BAGFILE)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Preproceso visual (para ver mejor)
    # Convertimos a gris usando solo canal verde para el estéreo
    gray_l = img_l[:,:,1] 
    gray_r = img_r[:,:,1]

    cv2.namedWindow('Stereo Tuner', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Stereo Tuner', 1000, 800)

    # Crear Trackbars
    # Num Disparities debe ser divisible por 16. Aquí usaremos bloques de 16.
    # Valor 1 = 16 disparidades, Valor 10 = 160.
    cv2.createTrackbar('Num Disps (x16)', 'Stereo Tuner', 8, 20, nothing) # Default 8*16=128
    cv2.createTrackbar('Min Disp', 'Stereo Tuner', 0, 100, nothing) # Puede ser negativo en código
    cv2.createTrackbar('Block Size', 'Stereo Tuner', 5, 21, nothing) # Debe ser impar
    cv2.createTrackbar('Uniqueness', 'Stereo Tuner', 10, 50, nothing)
    cv2.createTrackbar('Speckle Win', 'Stereo Tuner', 100, 200, nothing)
    cv2.createTrackbar('Speckle Rng', 'Stereo Tuner', 32, 50, nothing)
    cv2.createTrackbar('Disp12MaxDiff', 'Stereo Tuner', 1, 100, nothing)

    print("Iniciando GUI. Pulsa 'q' para salir.")

    while True:
        
        # Leer valores
        n_disp_mult = cv2.getTrackbarPos('Num Disps (x16)', 'Stereo Tuner')
        min_disp = cv2.getTrackbarPos('Min Disp', 'Stereo Tuner')
        block_size = cv2.getTrackbarPos('Block Size', 'Stereo Tuner')
        uniqueness = cv2.getTrackbarPos('Uniqueness', 'Stereo Tuner')
        speckle_win = cv2.getTrackbarPos('Speckle Win', 'Stereo Tuner')
        speckle_rng = cv2.getTrackbarPos('Speckle Rng', 'Stereo Tuner')
        disp12max = cv2.getTrackbarPos('Disp12MaxDiff', 'Stereo Tuner')

        # Validaciones para evitar crash de OpenCV
        if n_disp_mult < 1: n_disp_mult = 1
        num_disp = n_disp_mult * 16
        
        if block_size % 2 == 0: block_size += 1 # Debe ser impar
        if block_size < 5: block_size = 5

        # Crear Matcher con valores actuales
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            disp12MaxDiff=disp12max,
            uniquenessRatio=uniqueness,
            speckleWindowSize=speckle_win,
            speckleRange=speckle_rng,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Calcular Disparidad
        disp = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

        # Visualización
        # Normalizamos para ver algo (0 a num_disp) -> (0 a 255)
        disp_vis = (disp - min_disp) / num_disp
        disp_vis = np.clip(disp_vis, 0, 1)
        disp_vis = (disp_vis * 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

        # Máscara de zonas inválidas (negro)
        invalid_mask = (disp < min_disp)
        disp_color[invalid_mask] = 0

        # Mostrar Imagen original + Disparidad
        combined = np.hstack((img_l, disp_color))
        cv2.imshow('Stereo Tuner', combined)

        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            print(f"\n--- VALORES FINALES ---")
            print(f"minDisparity: {min_disp}")
            print(f"numDisparities: {num_disp}")
            print(f"blockSize: {block_size}")
            print(f"uniquenessRatio: {uniqueness}")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    tuner()