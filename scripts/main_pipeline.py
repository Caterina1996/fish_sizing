#!/usr/bin/env python3.9
import argparse
import os
import sys
import cv2
import yaml
import shutil
import numpy as np
from termcolor import cprint
from natsort import natsorted

from fish_sizing.bag_tools.bag_processor import BagProcessor
from fish_sizing.img_processing.image_processor import ImageProcessor
from fish_sizing.detection.fish2D import Fish2D, FrameScene  
from fish_sizing.stereo.stereo import StereoVision
from fish_sizing.detection.fish_detector import  FishDetector
from fish_sizing.measurement.fish3D import  Fish3D
from fish_sizing.analysis.bagfile_fauna import  Bagfile_fauna
from fish_sizing.measurement.fish_sizer import FishSizer

# --- CONFIGURACIÓN ---
PATH_MAPPINGS = {
    "/home/slimbook/bagfiles": "/home/rosuser/dataset/bagfiles",
    "/home/slimbook/fish_sizing/out": "/home/rosuser/repo/out",
    "/home/slimbook/models": "/home/rosuser/dataset/models/",
    "/home/slimbook/fish_sizing/config" :"/home/rosuser/repo/config/",
    "/media/slimbook/easystore": "/home/rosuser/easystore"
}

USE_DOCKER = True

TOPICS_DICT = { 
    "left":   "/stereo_ch3/left/image_raw",
    "right":  "/stereo_ch3/right/image_raw", 
    "info_l": "/stereo_ch3/left/camera_info",
    "info_r": "/stereo_ch3/right/camera_info"
}

BAGFILE_PATH=""
MODEL_PATH="/home/slimbook/models/binary/yv11m/Pool_v5-revisada_no_duplicats_from_ckpt/40e_finetune_2/weights/last.pt"
IN_DIR ="/media/slimbook/easystore/results_fish_sizing/seleccio_article/1_peix/2024_11_12/10_44_11/0/original_images/"
CAMERA_INFO_YAML_PATH ="/media/slimbook/easystore/results_fish_sizing/seleccio_article/Lanty_2/2025_08_20"

OUT_PATH = "/media/slimbook/easystore/results_fish_sizing/seleccio_article/1_peix/2024_11_12/10_44_11/0/processing2/"

CONF_THR = 0.5
gt =  33.5
Visualize_online = False
use_wls = True
image_channels = 3

SELECTED_PIPELINE = "dehazing"

# --- CONFIGURACIÓN DE PIPELINES ---
PROCESSING_PIPELINES = {
    "basic": [
    #     ("match_brightness_linear", {"reference": "left"}, False),
    #     ("convert_to_custom_grayscale", {}, True),
    #    ("apply_clahe", {"clip_limit": 2.0, "grid_size": (8,8)}, True)    
       
        ("match_brightness_linear", {"reference": "left"}, False),
        ("convert_to_custom_grayscale", {"w_g":0.65, "w_b":0.25, "w_r":0.1}, True),
        ("apply_clahe", {"clip_limit": 2.5, "grid_size": (8,8)}, True)  
    ],

    "dehazing": [
        ("apply_dehaze",            {"omega": 0.85}, True), 
        ("match_brightness_linear", {"reference": "left"}, False),
        ("apply_clahe",             {"clip_limit": 2.0}, True),
        ("match_histograms", {"reference": "left"}, True),
    ]
}

# --- CLASES Y FUNCIONES AUXILIARES ---

class CameraInfoMsg:
    """Clase auxiliar para simular un mensaje CameraInfo de ROS a partir de un diccionario YAML."""
    def __init__(self, yaml_data, scale=1.0):
        # 1. Escalar dimensiones
        self.width = int(yaml_data.get('image_width', 0) * scale)
        self.height = int(yaml_data.get('image_height', 0) * scale)
        
        # 2. Escalar Matriz Intrínseca (K) [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        K = yaml_data.get('camera_matrix', {}).get('data', [])
        self.K = [v * scale if i in [0, 2, 4, 5] else v for i, v in enumerate(K)]
        
        # 3. La distorsión y rectificación (D y R) NO se escalan
        self.D = yaml_data.get('distortion_coefficients', {}).get('data', [])
        self.R = yaml_data.get('rectification_matrix', {}).get('data', [])
        
        # 4. Escalar Matriz de Proyección (P) [fx', 0, cx', Tx, 0, fy', cy', Ty, 0, 0, 1, 0]
        P = yaml_data.get('projection_matrix', {}).get('data', [])
        self.P = [v * scale if i in [0, 2, 3, 5, 6, 7] else v for i, v in enumerate(P)]
        
        self.distortion_model = yaml_data.get('distortion_model', 'plumb_bob')

def transform_path2docker(path: str) -> str:
    if not USE_DOCKER or path is None:
        return path
    for host_path, docker_path in PATH_MAPPINGS.items():
        if host_path in path:
            new_path = path.replace(host_path, docker_path)
            cprint(f"🔄 Path mapped: {path} \n   -> {new_path}", "yellow")
            return new_path 
    return path

def stream_stereo_from_folder(folder_path):
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    all_files = os.listdir(folder_path)
    left_files = [f for f in all_files if "left" in f and f.lower().endswith(valid_exts)]
    left_files = natsorted(left_files)
    
    print(f"📂 Encontrados {len(left_files)} pares de imágenes en {folder_path}")

    for f_left in left_files:
        f_right = f_left.replace("left", "right")
        path_l = os.path.join(folder_path, f_left)
        path_r = os.path.join(folder_path, f_right)
        
        if not os.path.exists(path_r):
            print(f"⚠️ Aviso: No se encontró la pareja derecha para {f_left}. Saltando.")
            continue
            
        img_l = cv2.imread(path_l)
        img_r = cv2.imread(path_r)
        
        if img_l is None or img_r is None:
            print(f"❌ Error leyendo imágenes: {f_left}")
            continue
            
        frame_id_simulated = f_left.split(".")[0] 
        yield frame_id_simulated, img_l, img_r

def load_camera_info_from_yaml(folder_path, scale=1.0):
    """Carga left.yaml y right.yaml y los convierte en objetos compatibles con el pipeline, escalando si es necesario."""
    left_yaml_path = os.path.join(folder_path, 'left.yaml')
    right_yaml_path = os.path.join(folder_path, 'right.yaml')
    
    if not os.path.exists(left_yaml_path) or not os.path.exists(right_yaml_path):
        return {'left': None, 'right': None}
        
    with open(left_yaml_path, 'r') as f_l, open(right_yaml_path, 'r') as f_r:
        left_data = yaml.safe_load(f_l)
        right_data = yaml.safe_load(f_r)
        
    return {
        'left': CameraInfoMsg(left_data, scale=scale),
        'right': CameraInfoMsg(right_data, scale=scale)
    }

def save_run_config(out_dir, args):
    """Guarda toda la configuración de la ejecución en un archivo YAML para reproducibilidad."""
    stereo_cfg = {}
    if os.path.exists(args.stereo_config):
        with open(args.stereo_config, 'r') as f:
            stereo_cfg = yaml.safe_load(f)
            
    globals_cfg = {
        "MODEL_PATH": args.model_path,
        "CONF_THR": CONF_THR,
        "gt_ground_truth": gt,
        "Visualize_online": Visualize_online,
        "use_wls": use_wls,
        "image_channels": image_channels,
        "selected_pipeline_name": args.selected_pipeline
    }
    
    img_pipeline_steps = PROCESSING_PIPELINES.get(args.selected_pipeline, [])
    pipeline_readable = [{"step": step[0], "params": step[1], "enabled": step[2]} for step in img_pipeline_steps]

    full_config = {
        "execution_args": vars(args),
        "global_variables": globals_cfg,
        "image_processing_pipeline": pipeline_readable,
        "stereo_configuration": stereo_cfg
    }
    
    config_path = os.path.join(out_dir, "run_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)
    cprint(f"📄 Archivo de configuración guardado en: {config_path}", "green")


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description="Script para procesar bagfiles o carpetas y guardar imágenes filtradas.")
    parser.add_argument("--bag_file", "-bg", type=str, help="Ruta al bagfile", default=BAGFILE_PATH)
    parser.add_argument("--out_path", "-out", type=str, help="Carpeta donde se guardarán las procesadas", default=OUT_PATH)
    parser.add_argument("--images_source_dir", type=str, default=IN_DIR, help="Lee imágenes de esta carpeta en lugar del bagfile")
    parser.add_argument("--camera_info_yaml", type=str, default=CAMERA_INFO_YAML_PATH, help="Lee imágenes de esta carpeta en lugar del bagfile")
    
    # --- NUEVOS ARGUMENTOS PARA IMÁGENES PREPROCESADAS ---
    parser.add_argument("--is_preprocessed", action="store_true", default=True, 
                        help="Activar si las imágenes de la carpeta ya están rectificadas y decimadas.")
    parser.add_argument("--pre_scale", type=float, default=0.5, 
                        help="Factor de escala que YA tienen las imágenes guardadas respecto al YAML original (ej. 0.5)")
    
    parser.add_argument("--topic", type=str, default="left", choices=["left", "right"],help="Qué cámara exportar")
    parser.add_argument("--model_path", type=str,  help="Path to the detection AI model", default=MODEL_PATH)
    parser.add_argument("--stereo_config", type=str, default="/home/slimbook/fish_sizing/config/stereo_config.yaml", help="Path to the yaml con la config")
    parser.add_argument("--decimation", type=float, default=0.5, help="Escala de reduccion")
    parser.add_argument("--ignore_borders", action="store_true", default=True, help="Procesa peces aunque toquen bordes")
    parser.add_argument("--ignore_completeness", action="store_true", default=True, help="Procesa peces incompletos")
    parser.add_argument("--ignore_overlap", action="store_true", default=True, help="Procesa peces solapados")
    parser.add_argument("--overlap_margin", type=float, default=0.05, help="Margen Z en metros para solapamientos")
    parser.add_argument("--selected_pipeline", default=SELECTED_PIPELINE, help="Pipeline de procesamiento")
       
    args = parser.parse_args()
                                                            
    # 1. Transformar rutas para Docker
    bag_file = transform_path2docker(args.bag_file)
    out_path = transform_path2docker(args.out_path)
    model_path = transform_path2docker(args.model_path)
    stereo_config_path = transform_path2docker(args.stereo_config)  
    
    args.out_path = out_path
    args.model_path = model_path
    args.stereo_config = stereo_config_path
    
    # Lógica de escala
    if args.images_source_dir and args.is_preprocessed:
        # Si ya están decimadas, forzamos decimation=1.0 para este pipeline
        decimation = 1.0
        args.decimation = 1.0 
        scale_for_yaml = args.pre_scale # Escalar el YAML para que coincida con la imagen física
    else:
        decimation = args.decimation
        scale_for_yaml = 1.0

    # Crear carpeta de salida y guardar configuración
    os.makedirs(out_path, exist_ok=True)
    save_run_config(out_path, args)

    # 2. Inicializar fuente de datos y calibración
    if args.images_source_dir:
        # A) Modo Carpeta
        
        source_dir = transform_path2docker(args.images_source_dir)
        cprint(f"📂 Modo Carpeta: Leyendo imágenes y calibración desde {source_dir}", "cyan")
        
        # Leemos los yaml y le aplicamos el factor de escala (0.5) matemáticamente a las matrices
        if args.camera_info_yaml!="":
            
            camera_info_source_dir = transform_path2docker(args.camera_info_yaml)
        else:
            camera_info_source_dir = source_dir
        
        camera_info = load_camera_info_from_yaml(camera_info_source_dir, scale=scale_for_yaml)
        
        
        if camera_info['left'] is None or camera_info['right'] is None:
            cprint("❌ Error: No se encontraron left.yaml y right.yaml en la carpeta fuente.", "red")
            sys.exit(1)
            
        shutil.copy(os.path.join(source_dir, 'left.yaml'), os.path.join(out_path, 'left.yaml'))
        shutil.copy(os.path.join(source_dir, 'right.yaml'), os.path.join(out_path, 'right.yaml'))
        cprint(f"✅ Calibración YAML cargada y adaptada a factor {scale_for_yaml}x.", "green")
        
        image_iterator = stream_stereo_from_folder(source_dir)

    else:
        # B) Modo Bagfile
        cprint(f"📂 Modo Bagfile: Leyendo {bag_file}", "cyan")
        if not os.path.exists(bag_file):
            cprint(f"❌ Error: El archivo bag no existe: {bag_file}", "red")
            sys.exit(1)
            
        bag_proc = BagProcessor(bag_file, TOPICS_DICT)
        cprint("🔍 Buscando mensajes de calibración...", "yellow")
        camera_info = bag_proc.get_calibration()
        
        if camera_info['left'] is None or camera_info['right'] is None:
            cprint("❌ Error: No se encontró info de calibración en el bag. No se puede rectificar.", "red")
            sys.exit(1)
            
        bag_proc.save_calibration_yaml(out_path)
        cprint("✅ Calibración ROS extraída y guardada.", "green")
        
        image_iterator = bag_proc.stream_stereo_pairs()

    # 3. Inicializar ImageProcessor y detector
    img_proc = ImageProcessor(
        info_l=camera_info['left'], 
        info_r=camera_info['right']
    )
    
    fish_detector = FishDetector(model_path,conf_thr=CONF_THR)
    bagfile_fauna = Bagfile_fauna(out_path,gt)
    stereo = StereoVision(calibration_data=camera_info, config_path=stereo_config_path,scale=decimation)
    
    # 4. Bucle de Procesamiento
    cprint(f"🚀 Iniciando procesamiento y exportación a: {out_path}", "cyan")
    count = 0
    
    for timestamp, img_l_raw, img_r_raw in image_iterator:
        fname = f"frame_{count}"
        clean_name = str(count)
    
        img_proc.set_image_pair(img_l_raw, img_r_raw)
        
        # Lógica de pre-procesamiento
        if args.images_source_dir and args.is_preprocessed:
            # Las imágenes ya están rectificadas y decimadas. Nos saltamos ese paso.
            img_proc.processed_l = img_l_raw
            img_proc.processed_r = img_r_raw
        else:
            # Flujo normal para bagfiles o imágenes sin tratar
            img_proc.rectify()
            if count == 0: 
                cprint("🛠️ Abriendo verificador de rectificación...", "yellow")
                img_proc.check_rectification_interactive()
            img_proc.downsample(decimation)
            
        img_l, img_r = img_proc.get_processed()
               
        any_fish, frame_scene = fish_detector.process_frame(
            img_proc.processed_left, 
            frame_id=count, 
            disparity_img=None, 
            save_debug=True, 
            debug_path=out_path, 
            save_obj=True
        )
        
        if any_fish:
            processed_l, processed_r = img_proc.get_processed()
            
            cv2.imwrite(os.path.join(out_path, fname+"_original_left.png"), processed_l)
            cv2.imwrite(os.path.join(out_path, fname+"_original_right.png"), processed_r)
            
            processed_l, processed_r = img_proc.run_pipeline(
                PROCESSING_PIPELINES[args.selected_pipeline], 
                base_debug_folder = out_path,
                frame_id = fname,
                visualize = Visualize_online
            )

            cv2.imwrite(os.path.join(out_path, fname+"_left.png"), processed_l)
            cv2.imwrite(os.path.join(out_path, fname+"_right.png"), processed_r)
            
            strips = frame_scene.get_optimization_strips()
            disparity_map = stereo.compute_disparity(
                                    frame_id = fname,
                                    img_l = processed_l, 
                                    img_r =processed_r, 
                                    strips = strips, 
                                    use_wls=use_wls, 
                                    debug=False, 
                                    debug_path = out_path)
    
            frame_scene.disparity_map = disparity_map
            scene_points_3d = stereo.reproject_to_3d(disparity_map)     
            frame_scene.scene_points_3d = scene_points_3d
            
            valid_disp_mask = (disparity_map > stereo.min_valid_disparity)
            
            fish_sizer = FishSizer(frame_scene = frame_scene,
                                   img_l = img_l,
                                   bagfile_fauna = bagfile_fauna,
                                   stereo =stereo,
                                   args = args)
            
            bagfile_fauna = fish_sizer.measure_fish()
            
            scene_pcd = stereo.extract_point_cloud(scene_points_3d, img_l, mask=valid_disp_mask)
            
            frame_dir = os.path.join(out_path, f"{clean_name}")
            os.makedirs(frame_dir, exist_ok=True)
            scene_ply_name = os.path.join(frame_dir, f"{fname}_scene.ply")
            
            stereo.save_point_cloud(scene_pcd,save_path=scene_ply_name)
        
        else:
            cprint(f"No fish found in frame {count} :(, gonna process next image!","yellow")
        
        count += 1
        print(f"Procesado frame par: {count}", end='\r')
        
    bagfile_fauna.process_and_save_df()
    cprint(f"\n✅ Terminado. {count} pares guardados en {out_path}", "green")

if __name__ == "__main__":
    main()