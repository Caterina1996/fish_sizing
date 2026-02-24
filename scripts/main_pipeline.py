#!/usr/bin/env python3.9
import argparse
import os
import sys
import cv2
import numpy as np
from termcolor import cprint
from natsort import natsorted

from fish_sizing.utils.bag_processor import BagProcessor
from fish_sizing.utils.image_processor import ImageProcessor
from fish_sizing.detection.fish2D import Fish2D, FrameScene  # pot ser aquest import ens el podem estalviar
from fish_sizing.stereo.stereo import StereoVision
from fish_sizing.detection.fish_detector import  FishDetector
from fish_sizing.measurement.fish3D import  Fish3D
from fish_sizing.analysis.bagfile_fauna import  Bagfile_fauna
from fish_sizing.analysis.fish_sizer import FishSizer


# --- CONFIGURACIÓN ---
PATH_MAPPINGS = {
    "/home/slimbook/bagfiles": "/home/rosuser/dataset/bagfiles",
    "/home/slimbook/fish_sizing/out": "/home/rosuser/repo/out",
    "/home/slimbook/models": "/home/rosuser/dataset/models/",
    "home/slimbook/fish_sizing/config" :"/home/rosuser/repo/config/"
}

USE_DOCKER = True

TOPICS_DICT = { 
    "left":   "/stereo_ch3/left/image_raw",
    "right":  "/stereo_ch3/right/image_raw", 
    "info_l": "/stereo_ch3/left/camera_info",
    "info_r": "/stereo_ch3/right/camera_info"
}

# BAGFILE_PATH="//home/slimbook/bagfiles/LIMA/2025/2025_08_21/test_comprsesion/compressed/13_34_24/stereo_camera_images_2025-08-21-13-34-25_0_compressed.bag"

BAGFILE_PATH="/home/slimbook/bagfiles/Escenaris/Escenari_1/2024_11_28/13_07_38/stereo_camera_images_2024-11-28-13-07-39_0.bag"

# model_path="/home/slimbook/yolov8/trained_models/fish_detector.pt"
# model_path="/home/slimbook/models/Segmentation/pool/last_pool_nano_binary.pt"
# model_path="/home/slimbook/models/Segmentation/pool/25ckpt+POOL_y11_large/last.pt"
# model_path="//home/slimbook/models/Segmentation/pool/yv11l_25ckpt+pool_new/weights/last.pt"

# MODEL_PATH="/home/slimbook/models/25c_ckpt+PISCINA_NEW/yv11l_25ckpt+pool_new/weights/last.pt" #-> Provar aquest!!

MODEL_PATH="/home/slimbook/models/binary/yv11m/yv11m_binary_Pool_revisada_no_duplicada_from scractch/weights/best.pt"
# MODEL_PATH = "/home/slimbook/models/yv11l/ylarge_d18_poolv2r_lantytr_nocturnes/weights/best.pt"
CONF_THR = 0.5

gt = 28.9
# gt =None
Visualize_online = False


# peix/marca	t_tot	t_std
# vermella	    29,1	25,3
# verda	        29,2	25,8
# negra	        26,6	23,4
# sense	        32,3	28,7


# OUT_PATH = "/home/slimbook/fish_sizing/out/test_export/2025-05-08-11-18-25_1/"
OUT_PATH = "/home/slimbook/fish_sizing/out/Escenaris/Escenari_1/2024_11_28/13_07_38/"
# IN_PATH = "/home/slimbook/fish_sizing/out/Llobarros/2024_11_27/12_00_27/"

SELECTED_PIPELINE = "basic"

# --- CONFIGURACIÓN DE PIPELINES ---
PROCESSING_PIPELINES = {
    "basic": [
        # 1. Igualar luz
        ("match_brightness_linear", {"reference": "left"}, False),
        
        # 2. Convertir a gris ignorando el rojo
        ("convert_to_custom_grayscale", {}, True),
        
        # 3. CLAHE 
       ("apply_clahe", {"clip_limit": 2.0, "grid_size": (8,8)}, True)      
    ],

    "dehazing": [
        ("apply_dehaze",            {"omega": 0.85}, True), 
        ("match_brightness_linear", {"reference": "left"}, False),
        # ("convert_to_custom_grayscale", {}, True),
        ("apply_clahe",             {"clip_limit": 2.0}, True),
        ("match_histograms", {"reference": "left"}, True),
    ]
}


# --- FUNCIONES AUXILIARES ---

def transform_path2docker(path: str) -> str:
    """Transform a path from local computer to docker structure."""
    if not USE_DOCKER or path is None:
        return path

    for host_path, docker_path in PATH_MAPPINGS.items():
        if host_path in path:
            new_path = path.replace(host_path, docker_path)
            cprint(f"🔄 Path mapped: {path} \n   -> {new_path}", "yellow")
            return new_path 
            
    return path

def stream_stereo_from_folder(folder_path):
    """
    Generador que lee pares de imágenes (left/right) desde una carpeta.
    Espera nombres tipo: '...original_left.png' y '...original_right.png'
    """
    # Extensiones válidas
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    
    # 1. Listar archivos y filtrar solo los LEFT
    all_files = os.listdir(folder_path)
    left_files = [f for f in all_files if "left" in f and f.lower().endswith(valid_exts)]
    
    # 2. Ordenar naturalmente (1, 2, ... 10)
    left_files = natsorted(left_files)
    
    print(f"📂 Encontrados {len(left_files)} pares de imágenes en {folder_path}")

    for f_left in left_files:
        # Construir el nombre del archivo RIGHT asumiendo simetría en el nombre
        # Ejemplo: frame_0_original_left.png -> frame_0_original_right.png
        f_right = f_left.replace("left", "right")
        
        path_l = os.path.join(folder_path, f_left)
        path_r = os.path.join(folder_path, f_right)
        
        # Verificar que existe la pareja derecha
        if not os.path.exists(path_r):
            print(f"⚠️ Aviso: No se encontró la pareja derecha para {f_left}. Saltando.")
            continue
            
        # Cargar imágenes
        img_l = cv2.imread(path_l)
        img_r = cv2.imread(path_r)
        
        if img_l is None or img_r is None:
            print(f"❌ Error leyendo imágenes: {f_left}")
            continue
            
        # Usamos el nombre del archivo como 'timestamp' o ID para mantener coherencia
        frame_id_simulated = f_left.split(".")[0] 
        
        # Yield (devuelve los valores uno a uno, igual que el bag_proc)
        yield frame_id_simulated, img_l, img_r



# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Script para procesar bagfiles y guardar imágenes filtradas.")
    
    parser.add_argument("--bag_file", "-bg", type=str, help="Ruta al bagfile",
                        default=BAGFILE_PATH)

    parser.add_argument("--out_path", "-out", type=str, help="Carpeta donde se guardarán las imágenes procesadas",
                        default=OUT_PATH)
    
    parser.add_argument("--images_source_dir", type=str, default=None,
                        help="Si se especifica, lee imágenes de esta carpeta en lugar del bagfile")
    
    parser.add_argument("--topic", type=str, default="left", choices=["left", "right"],help="Qué cámara exportar")
    
    parser.add_argument("--model_path", type=str,  help="Path to the detection AI model",
                        default=MODEL_PATH)
                        
    parser.add_argument("--stereo_config", type=str, default="/home/slimbook/fish_sizing/config/stereo_config.yaml", 
                        help="Path to the yaml with the config for the stereo matching alg")
    
    parser.add_argument("--decimation", type=float, default=0.5, help="reescale the images")
    
    parser.add_argument("--ignore_borders", action="store_true", default=True,
                        help="Si se activa, procesa peces aunque toquen los bordes de la imagen")
    
    parser.add_argument("--ignore_completeness", action="store_true", default=True,
                        help="Si se activa, procesa peces aunque falten datos 3D (incomplete)")
    
    parser.add_argument("--ignore_overlap", action="store_true", default=True,
                        help="Si se activa, procesa peces aunque los peces se solapen entre si")
    
    parser.add_argument("--overlap_margin", type=float, default=0.05,
                        help="Margen en metros para decidir qué pez está delante en solapamientos (Def: 0.05m)")
    
    parser.add_argument("--selected_pipeline", default=SELECTED_PIPELINE,
                        help="Pipeline de procesamiento de imagenes")
       
    args = parser.parse_args()
    
                                                            
    # 1. Transformar rutas para Docker
    bag_file = transform_path2docker(args.bag_file)
    out_path = transform_path2docker(args.out_path)
    model_path = transform_path2docker(args.model_path)
    stereo_config_path = transform_path2docker(args.stereo_config)  
    
    args.out_path = out_path
    args.model_path = model_path
    args.stereo_config = stereo_config_path

    
    decimation = args.decimation
    save_scene_pc = True

    if not os.path.exists(bag_file):
        cprint(f"❌ Error: El archivo bag no existe: {bag_file}", "red")
        sys.exit(1)

    # 2. Inicializar BagProcessor
    cprint(f"📂 Leyendo bag: {bag_file}", "cyan")
    bag_proc = BagProcessor(bag_file, TOPICS_DICT)

    # 3. Obtener Calibración (Necesaria para rectificar)
    cprint("🔍 Buscando mensajes de calibración...", "yellow")
    camera_info = bag_proc.get_calibration()
    
    if camera_info['left'] is None or camera_info['right'] is None:
        cprint("❌ Error: No se encontró info de calibración en el bag. No se puede rectificar.", "red")
        sys.exit(1)
        
    cprint("✅ Calibración encontrada.", "green")
    
    bag_proc.save_calibration_yaml(out_path)
    
    cprint("✅ Calibración guardada.", "green")

    # 4. Inicializar ImageProcessor (con la info de calibración) i el detector
    img_proc = ImageProcessor(
        info_l=camera_info['left'], 
        info_r=camera_info['right']
    )
    
    fish_detector = FishDetector(model_path,conf_thr=CONF_THR)
    bagfile_fauna = Bagfile_fauna(out_path,gt)
    
    stereo = StereoVision(calibration_data=camera_info, config_path=stereo_config_path,scale=decimation)

    f = stereo.FOCAL
    B = stereo.BASELINE
    
    
    # 5. Bucle de Procesamiento (Stereo Stream)
    cprint(f"🚀 Iniciando procesamiento y exportación a: {out_path}", "cyan")
    
    count = 0
    
    # Definir el iterador (fuente de datos)
    if args.images_source_dir:
        # A) Modo Carpeta
        source_dir = transform_path2docker(args.images_source_dir)
        cprint(f"📂 Modo Carpeta: Leyendo desde {source_dir}", "cyan")
        image_iterator = stream_stereo_from_folder(transform_path2docker(source_dir))
    else:
        # B) Modo Bagfile (Tu código original)
        cprint(f"📂 Modo Bagfile: Leyendo {bag_file}", "cyan")
        image_iterator = bag_proc.stream_stereo_pairs()
    
    
    # Usamos stream_stereo_pairs porque necesitamos AMBAS imágenes para rectificar
    for timestamp, img_l_raw, img_r_raw in image_iterator:
    
        # if count > 20:
        #     break
        
        # frame name TODO: decidir si vull el timestamp o count per facilitat
        # fname = f"{timestamp}"
        fname = f"frame_{count}"
    
        # Load the stereo pair
        img_proc.set_image_pair(img_l_raw, img_r_raw)
        
        # 1. Rectificar and downsample (we will use decimated_x2 images)
        img_proc.rectify()
        
        if count == 0: # Sanity check
            cprint("🛠️ Abriendo verificador de rectificación...", "yellow")
            img_proc.check_rectification_interactive()
        
        img_proc.downsample(decimation)
        img_l, img_r = img_proc.get_processed()
               
        # 2. Look for fish in the scene
        any_fish, frame_scene = fish_detector.process_frame(img_proc.processed_left, 
                                    frame_id=count, # decide weather to use this or timestamp 
                                    disparity_img=None, 
                                    save_debug=True, 
                                    debug_path=out_path, 
                                    save_obj=True)
        if any_fish:
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 1. Process image to improve the stereo matching later then 
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # EJECUTAR PIPELINE
            
            processed_l, processed_r = img_proc.get_processed()
            
            cv2.imwrite(os.path.join(out_path, fname+"_original_left.png"), processed_l)
            cv2.imwrite(os.path.join(out_path, fname+"_original_right.png"), processed_r)
            
            processed_l, processed_r = img_proc.run_pipeline(
                PROCESSING_PIPELINES[args.selected_pipeline], 
                base_debug_folder = out_path,
                frame_id = fname,
                visualize = Visualize_online
            )

            # SAVE IMAGES IF THEY CONTAIN FISH
            cv2.imwrite(os.path.join(out_path, fname+"_left.png"), processed_l)
            cv2.imwrite(os.path.join(out_path, fname+"_right.png"), processed_r)
            
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 2. Get strips for the calculation of the disparity + calculate disp and obtain pc
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            strips = frame_scene.get_optimization_strips()

            disparity_map = stereo.compute_disparity(
                                    frame_id = fname,
                                    img_l = processed_l, 
                                    img_r =processed_r, 
                                    strips = strips, 
                                    use_wls=True, 
                                    debug=False, 
                                    debug_path = out_path)
    
            # Inyectar disparidad en la escena y validar peces
            frame_scene.disparity_map = disparity_map
           

            # 2.1 Reproyectar toda la imagen a XYZ
            scene_points_3d = stereo.reproject_to_3d(disparity_map)     
            frame_scene.scene_points_3d = scene_points_3d
            
            # Inicializar la máscara para seleccionar el trozo de pc que queremos
            all_pc_mask = np.zeros(disparity_map.shape, dtype=bool)
           
            
            # Filtrar las distancias > dist__max (no me fio de mesures més enfora de 4m)
            valid_disp_mask = (disparity_map > stereo.min_valid_disparity)
            
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 3. Measure Fish
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            fish_sizer = FishSizer(frame_scene = frame_scene,
                                   img_l = img_l,
                                   bagfile_fauna = bagfile_fauna,
                                   stereo =stereo,
                                   args = args)
            
            bagfile_fauna = fish_sizer.measure_fish()
            
            # Save scene
            scene_pcd = stereo.extract_point_cloud(scene_points_3d, img_l, mask=valid_disp_mask)
            scene_ply_name = os.path.join(out_path, f"{fname}_scene.ply")
            stereo.save_point_cloud(scene_pcd,save_path=scene_ply_name)
        
        else:
            cprint(f"No fish found in frame {count} :(, gonna process next image!","yellow")
        
        count += 1
        print(f"Procesado frame par: {count}", end='\r')
        
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 4. Log and process bagfile fauna
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    
    bagfile_fauna.process_and_save_df()

    cprint(f"\n✅ Terminado. {count} pares guardados en {out_path}", "green")

if __name__ == "__main__":
    main()

