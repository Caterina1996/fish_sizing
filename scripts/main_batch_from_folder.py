#!/usr/bin/env python3.9
import argparse
import os
import sys
import cv2
import shutil
import numpy as np
from pathlib import Path
from termcolor import cprint
from natsort import natsorted
import yaml

from fish_sizing.bag_tools.bag_processor import BagProcessor
from fish_sizing.img_processing.image_processor import ImageProcessor
from fish_sizing.detection.fish2D import Fish2D, FrameScene  
from fish_sizing.stereo.stereo import StereoVision
from fish_sizing.detection.fish_detector import FishDetector
from fish_sizing.measurement.fish3D import Fish3D
from fish_sizing.analysis.bagfile_fauna import Bagfile_fauna
from fish_sizing.measurement.fish_sizer import FishSizer

# ==========================================
# --- CONFIGURACIÓN DE RUTAS POR DEFECTO ---
# ==========================================
# Rellena SOLO la ruta del modo que quieras usar. Deja las demás con ""

# 1. Modos de Entrada (El script ejecutará el primero que tenga texto)
IMAGES_BATCH_DIR = "/media/slimbook/easystore/results_fish_sizing/seleccio_article/lanty1/2025_08_21/multiple_fish" # Modo Batch Imágenes Extraídas
BAGS_DIR         = "" # "/media/slimbook/easystore/bagfiles/LIMA/2025/Lanty_2/2025_08_21/"
IMAGES_SOURCE_DIR= "" # "/ruta/a/una/sola/carpeta"
BAGFILE_PATH     = "" # "/ruta/a/un/solo.bag"

# 2. Configuración para Imágenes Extraídas (Modo Batch)
TARGET_IMG_FOLDER     = "original_images"
OUT_FOLDER_NAME       = "results_article_basic_nou"
CAMERA_INFO_YAML_PATH = "//media/slimbook/easystore/results_fish_sizing/seleccio_article/lanty1/2025_08_21/multiple_fish/"
IS_PREPROCESSED       = True  # True si las imágenes ya están decimadas/rectificadas
PRE_SCALE             = 0.5   # Factor de escala al que se guardaron las imágenes

# 3. Salida y Modelos (Para modos que no son Batch Imágenes)
OUT_PATH   = "//media/slimbook/easystore/results_fish_sizing/seleccio_article/lanty1/2025_08_21/multiple_fish/"
# MODEL_PATH = "/home/slimbook/models/binary/yv11m/Pool_v5-revisada_no_duplicats_from_ckpt/40e_finetune_2/weights/last.pt"
MODEL_PATH="/home/slimbook/models/binary/yv11m/Pool_v5-revisada_no_duplicats_from_ckpt/40e_finetune_2/weights/last.pt"

# 4. Parámetros del Algoritmo
CONF_THR          = 0.5
gt                = 0
Visualize_online  = False
use_wls           = True
image_channels    = 1 # Si USAMOS EL COLOR CAMBIAR A 3
SELECTED_PIPELINE = "basic"
overwrite_existing = True

# --- DOCKER MAPPINGS ---
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

# --- CONFIGURACIÓN DE PIPELINES ---
# --- CONFIGURACIÓN DE PIPELINES ---
PROCESSING_PIPELINES = {
    "basic": [
        ("match_brightness_linear", {"reference": "left"}, False),
        ("convert_to_custom_grayscale", {"w_g":0.65, "w_b":0.25, "w_r":0.1},False),
        ("apply_clahe", {"clip_limit": 2.0, "grid_size": (8,8)}, False)      
    ],
    
    # NUEVO PIPELINE 1: Textura extrema (Ideal para peces lisos)
    "sharp_stereo": [
        ("match_brightness_linear", {"reference": "left"}, False),
        ("apply_gamma",             {"gamma": 1.2}, True), # Levanta sombras suavemente
        ("apply_bilateral",         {"d": 7, "sigma_color": 50, "sigma_space": 50}, True), # Mata el ruido del agua
        ("apply_sharpen",           {"alpha": 1.7}, True), # Exagera escamas y aletas
        ("convert_to_custom_grayscale", {}, False),
        ("match_histograms",        {"reference": "left"}, True) # Asegura que izq y der sean idénticas
    ],
    
    # NUEVO PIPELINE 2: Solo bordes limpios (Ideal si hay mucha suciedad en el agua)
    "clean_edges": [
        ("apply_bilateral",         {"d": 9, "sigma_color": 75, "sigma_space": 75}, True), # Planchado agresivo
        ("convert_to_custom_grayscale", {}, False),
        ("apply_clahe",             {"clip_limit": 1.5}, True), # CLAHE más suave
        # ("match_histograms",        {"reference": "left"}, True)
    ],

    "dehazing": [
        ("apply_dehaze",            {"omega": 0.85}, True), 
        ("match_brightness_linear", {"reference": "left"}, False),
        ("convert_to_custom_grayscale", {}, False),
        ("apply_clahe",             {"clip_limit": 2.0}, True),
        ("match_histograms", {"reference": "left"}, True),
    ]
}

# --- CLASES Y FUNCIONES AUXILIARES ---
class CameraInfoMsg:
    def __init__(self, yaml_data, scale=1.0):
        self.width = int(yaml_data.get('image_width', 0) * scale)
        self.height = int(yaml_data.get('image_height', 0) * scale)
        K = yaml_data.get('camera_matrix', {}).get('data', [])
        self.K = [v * scale if i in [0, 2, 4, 5] else v for i, v in enumerate(K)]
        self.D = yaml_data.get('distortion_coefficients', {}).get('data', [])
        self.R = yaml_data.get('rectification_matrix', {}).get('data', [])
        P = yaml_data.get('projection_matrix', {}).get('data', [])
        self.P = [v * scale if i in [0, 2, 3, 5, 6, 7] else v for i, v in enumerate(P)]
        self.distortion_model = yaml_data.get('distortion_model', 'plumb_bob')

def transform_path2docker(path: str) -> str:
    if not USE_DOCKER or path is None:
        return path
    for host_path, docker_path in PATH_MAPPINGS.items():
        if host_path in path:
            new_path = path.replace(host_path, docker_path)
            return new_path 
    return path

def save_run_config(out_dir, args):
    stereo_cfg = {}
    if args.stereo_config and os.path.exists(args.stereo_config):
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
    pipeline_readable = [{"step": step[0], "params": step[1], "save_debug_image": step[2]} for step in img_pipeline_steps]

    full_config = {
        "execution_args": vars(args),
        "global_variables": globals_cfg,
        "image_processing_pipeline": pipeline_readable,
        "stereo_configuration": stereo_cfg
    }
    
    config_path = os.path.join(out_dir, "run_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)

def stream_stereo_from_folder(folder_path):
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    all_files = os.listdir(folder_path)
    left_files = [f for f in all_files if "left" in f and f.lower().endswith(valid_exts)]
    left_files = natsorted(left_files)
    
    print(f"📂 Encontrados {len(left_files)} pares en {folder_path}")

    for f_left in left_files:
        f_right = f_left.replace("left", "right")
        path_l = os.path.join(folder_path, f_left)
        path_r = os.path.join(folder_path, f_right)
        
        if not os.path.exists(path_r):
            continue
            
        img_l = cv2.imread(path_l)
        img_r = cv2.imread(path_r)
        
        if img_l is None or img_r is None:
            continue
            
        frame_id_simulated = f_left.split(".")[0] 
        yield frame_id_simulated, img_l, img_r

def move_inferred_images(out_path):
    inferred_dir = os.path.join(out_path, "_inferred")
    os.makedirs(inferred_dir, exist_ok=True)
    out_p = Path(out_path)
    moved_count = 0
    
    for file_path in out_p.rglob("*_inferred*.*"):
        if "_inferred" in file_path.parent.parts:
            continue
        if file_path.is_file():
            dest_path = os.path.join(inferred_dir, file_path.name)
            shutil.move(str(file_path), dest_path)
            moved_count += 1
            
    if moved_count > 0:
        cprint(f"✅ Movidas {moved_count} imágenes a _inferred/", "green")

def load_camera_info_from_yaml(folder_path, scale=1.0):
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

# --- CORE DEL PIPELINE ---
def process_single_source(source_path, current_out_path, is_dir, args):
    args.out_path = current_out_path
    os.makedirs(current_out_path, exist_ok=True)
    save_run_config(current_out_path, args)
    
    if is_dir and args.is_preprocessed:
        decimation = 1.0
        scale_for_yaml = args.pre_scale
    else:
        decimation = args.decimation
        scale_for_yaml = 1.0

    if not is_dir and not os.path.exists(source_path):
        cprint(f"❌ Error: El origen no existe: {source_path}", "red")
        return

    # 1. Inicializar Fuente de Datos 
    if not is_dir:
        cprint(f"📂 Leyendo bag: {source_path}", "cyan")
        topics_dict = TOPICS_DICT.copy()
        if "compressed" in os.path.basename(source_path).lower():
            cprint("🗜️ Modo comprimido detectado.", "yellow")
            topics_dict["left"] = "/stereo_ch3/left/image_raw/compressed"
            topics_dict["right"] = "/stereo_ch3/right/image_raw/compressed"
            
        bag_proc = BagProcessor(source_path, topics_dict)
        camera_info = bag_proc.get_calibration()
        
        if camera_info['left'] is None:
            cprint("❌ Error: No se encontró info de calibración.", "red")
            return
            
        bag_proc.save_calibration_yaml(current_out_path)
        image_iterator = bag_proc.stream_stereo_pairs()
    else:
        cprint(f"📂 Modo Carpeta: Leyendo desde {source_path}", "cyan")
        
        if args.camera_info_yaml and args.camera_info_yaml != "":
            yaml_source = transform_path2docker(args.camera_info_yaml)
        else:
            parent_dir = os.path.dirname(source_path)
            if os.path.exists(os.path.join(parent_dir, 'left.yaml')):
                yaml_source = parent_dir
            else:
                yaml_source = source_path
                
        camera_info = load_camera_info_from_yaml(yaml_source, scale=scale_for_yaml)
        
        if camera_info['left'] is None:
            cprint(f"❌ Error: No se encontraron left/right.yaml en {yaml_source}", "red")
            return
            
        try:
            shutil.copy(os.path.join(yaml_source, 'left.yaml'), os.path.join(current_out_path, 'left.yaml'))
            shutil.copy(os.path.join(yaml_source, 'right.yaml'), os.path.join(current_out_path, 'right.yaml'))
        except shutil.SameFileError:
            pass 
            
        image_iterator = stream_stereo_from_folder(folder_path=source_path)

    # 2. Inicializar Módulos
    img_proc = ImageProcessor(info_l=camera_info['left'], info_r=camera_info['right'])
    fish_detector = FishDetector(args.model_path, conf_thr=CONF_THR)
    bagfile_fauna = Bagfile_fauna(current_out_path, gt)
    stereo = StereoVision(calibration_data=camera_info, config_path=args.stereo_config, scale=decimation, image_channels=image_channels)
    
    # 3. Bucle de Procesamiento
    cprint(f"🚀 Iniciando procesamiento hacia: {current_out_path}", "cyan")
    count = 0
    
    for timestamp, img_l_raw, img_r_raw in image_iterator:
        fname = f"frame_{count}"
        clean_name = str(count) 
    
        img_proc.set_image_pair(img_l_raw, img_r_raw)
        
        if is_dir and args.is_preprocessed:
            img_proc.processed_left = img_l_raw
            img_proc.processed_right = img_r_raw
        else:
            img_proc.rectify()
            img_proc.downsample(decimation)
            
        img_l, img_r = img_proc.get_processed()
               
        any_fish, frame_scene = fish_detector.process_frame(
            img_proc.processed_left, 
            frame_id=fname, 
            disparity_img=None, 
            save_debug=True, 
            debug_path=current_out_path, 
            save_obj=True
        )
        
        if any_fish:
            processed_l, processed_r = img_proc.get_processed()
            
            processed_l, processed_r = img_proc.run_pipeline(
                PROCESSING_PIPELINES[args.selected_pipeline], 
                base_debug_folder = current_out_path,
                frame_id = fname,
                visualize = Visualize_online
            )

            cv2.imwrite(os.path.join(current_out_path, fname+"_left.png"), processed_l)
            cv2.imwrite(os.path.join(current_out_path, fname+"_right.png"), processed_r)
            
            strips = frame_scene.get_optimization_strips()
            disparity_map = stereo.compute_disparity(
                                    frame_id = fname,
                                    img_l = processed_l, 
                                    img_r =processed_r, 
                                    strips = strips, 
                                    use_wls=use_wls, 
                                    debug=False, 
                                    debug_path = current_out_path)
    
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
            frame_dir = os.path.join(current_out_path, f"frame_{clean_name}")
            os.makedirs(frame_dir, exist_ok=True)
            scene_ply_name = os.path.join(frame_dir, f"{clean_name}_scene.ply")
            
            stereo.save_point_cloud(scene_pcd, save_path=scene_ply_name)
        
        else:
            cprint(f"No fish found in frame {count}", "yellow")
        
        count += 1
        print(f"Procesado frame par: {count}", end='\r')
        
    bagfile_fauna.process_and_save_df()
    cprint(f"\n✅ Terminado. {count} frames procesados.", "green")

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description="Script para procesar bagfiles o carpetas extraídas.")
    
    # Orígenes de Datos (Elige uno)
    parser.add_argument("--images_batch_dir", type=str, default=IMAGES_BATCH_DIR)
    parser.add_argument("--bags_dir", "-bd", type=str, default=BAGS_DIR)
    parser.add_argument("--images_source_dir", type=str, default=IMAGES_SOURCE_DIR)
    parser.add_argument("--bag_file", "-bg", type=str, default=BAGFILE_PATH)
    
    # Nombres de carpetas para el modo Batch Imágenes
    parser.add_argument("--target_img_folder", type=str, default=TARGET_IMG_FOLDER)
    parser.add_argument("--out_folder_name", type=str, default=OUT_FOLDER_NAME)

    # Configuración de Calibración e Imágenes
    parser.add_argument("--camera_info_yaml", type=str, default=CAMERA_INFO_YAML_PATH)
    parser.add_argument("--is_preprocessed", action="store_true", default=IS_PREPROCESSED)
    parser.add_argument("--pre_scale", type=float, default=PRE_SCALE)
    
    # Salida Genérica y Modelos
    parser.add_argument("--out_path", "-out", type=str, default=OUT_PATH)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--stereo_config", type=str, default="/home/slimbook/fish_sizing/config/stereo_config.yaml")
    parser.add_argument("--selected_pipeline", default=SELECTED_PIPELINE)
    parser.add_argument("--decimation", type=float, default=0.5)
    
    # Checks para calcular el 3d
    parser.add_argument("--ignore_borders", action="store_true", default=False, help="Procesa peces aunque toquen bordes")
    parser.add_argument("--ignore_completeness", action="store_true", default=False, help="Procesa peces incompletos")
    parser.add_argument("--ignore_overlap", action="store_true", default=True, help="Procesa peces solapados")
    parser.add_argument("--overlap_margin", type=float, default=0.05, help="Margen Z en metros para solapamientos")
    
    # --- LÍNEAS QUE FALTABAN ---
    parser.add_argument("--ignore_aspect_ratio", action="store_true", default=False, help="Procesa peces cuadrados")
    parser.add_argument("--aspect_ratio_thr", type=float, default=2.0, help="Aspect ratio check thr para descartar peces frontales")
    
    args = parser.parse_args()
    
    args.model_path = transform_path2docker(args.model_path)
    args.stereo_config = transform_path2docker(args.stereo_config)
    
    tasks = [] 
    
    # Evaluamos en cascada asegurando que el string no esté vacío
    if args.images_batch_dir and args.images_batch_dir.strip() != "":
        batch_dir = transform_path2docker(args.images_batch_dir)
        cprint(f"🔍 [MODO 1] Escaneando carpetas batch en: {batch_dir}", "cyan")
        
        for root, dirs, files in os.walk(batch_dir):
            if os.path.basename(root) == args.target_img_folder:
                source_path = root
                parent_dir = os.path.dirname(root)
                specific_out = os.path.join(parent_dir, args.out_folder_name)
                tasks.append((source_path, specific_out, True))

    elif args.bags_dir and args.bags_dir.strip() != "":
        out_base = transform_path2docker(args.out_path)
        bags_dir = transform_path2docker(args.bags_dir)
        cprint(f"🔍 [MODO 2] Escaneando bagfiles en: {bags_dir}", "cyan")
        
        for root, _, files in os.walk(bags_dir):
            for file in files:
                if file.endswith(".bag"):
                    bag_path = os.path.join(root, file)
                    bag_name_no_ext = os.path.splitext(file)[0]
                    bag_suffix = bag_name_no_ext.split('_')[-1] 
                    rel_path = os.path.relpath(root, bags_dir)
                    
                    if rel_path == ".":
                        specific_out = os.path.join(out_base, bag_suffix)
                    else:
                        specific_out = os.path.join(out_base, rel_path, bag_suffix)
                        
                    tasks.append((bag_path, specific_out, False))
                    
    elif args.images_source_dir and args.images_source_dir.strip() != "":
        cprint(f"🔍 [MODO 3] Procesando carpeta simple: {args.images_source_dir}", "cyan")
        source = transform_path2docker(args.images_source_dir)
        out_base = transform_path2docker(args.out_path)
        tasks.append((source, out_base, True))
        
    elif args.bag_file and args.bag_file.strip() != "":
        cprint(f"🔍 [MODO 4] Procesando bagfile simple: {args.bag_file}", "cyan")
        source = transform_path2docker(args.bag_file)
        out_base = transform_path2docker(args.out_path)
        tasks.append((source, out_base, False))
        
    if not tasks:
        cprint("❌ No se encontraron datos para procesar. Verifica las variables globales de arriba.", "red")
        sys.exit(1)

    cprint(f"📋 Se procesarán {len(tasks)} orígenes en total.", "yellow")
    
    for i, (source_path, current_out_path, is_dir) in enumerate(tasks):
        cprint("\n" + "="*60, "magenta")
        cprint(f"🚀 TAREA {i+1}/{len(tasks)}", "magenta", attrs=['bold'])
        cprint(f"📄 Origen: {source_path}", "magenta")
        cprint(f"📁 Salida: {current_out_path}", "magenta")
        cprint("="*60 + "\n", "magenta")
        
        # --- NUEVA COMPROBACIÓN AQUÍ ---
        # Comprueba si la carpeta ya existe y además tiene algún archivo dentro
        if os.path.exists(current_out_path) and os.path.isdir(current_out_path) and len(os.listdir(current_out_path)) > 0:
            
            if overwrite_existing==False:
                cprint(f"⏭️  SALTANDO: La carpeta de salida ya existe y contiene datos.", "yellow")
                continue
            else:
               shutil.rmtree(current_out_path,ignore_errors=True)
        # -------------------------------
        
        try:
            process_single_source(source_path, current_out_path, is_dir, args)
            move_inferred_images(current_out_path)
            
        except Exception as e:
            cprint(f"❌ Error CRÍTICO procesando {source_path}: {e}", "red")

if __name__ == "__main__":
    main()