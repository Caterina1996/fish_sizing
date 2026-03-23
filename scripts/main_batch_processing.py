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
from fish_sizing.utils.config import transform_path2docker
from fish_sizing.utils.tools import save_run_config, cprint_and_log, move_inferred_images

BAGS_DIR="//media/slimbook/easystore1/bagfiles/seleccio_article/2025_08_21/lanty_1/1_peix/"
MODEL_PATH="/home/slimbook/models/binary/yv11m/Pool_v5-revisada_no_duplicats_from_ckpt/40e_finetune_2/weights/last.pt"

CONF_THR = 0.5
gt = ""
Visualize_online = False
use_wls = True

OUT_PATH = "//media/slimbook/easystore1/results_fish_sizing/seleccio_article/lanty1/2025_08_21/single_fish/"
SELECTED_PIPELINE = "basic"
image_channels = 1 # ATTENTION!!! CHANGE TO 3 FOR PIPELINES THAT USE COLOR IMAGES AS OUTPUT


def stream_stereo_from_folder(folder_path):
    """Generador que lee pares de imágenes desde una carpeta."""
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




# --- CORE DEL PIPELINE (Extraído del main original) ---
def process_single_source(source_path, current_out_path, is_dir, args):
    """Procesa un único bagfile o carpeta de imágenes"""
    
    args.out_path = current_out_path
    
    os.makedirs(current_out_path, exist_ok=True)
    decimation = args.decimation

    save_run_config(current_out_path, args)
    
    if not is_dir and not os.path.exists(source_path):
        cprint(f"❌ Error: El archivo bag no existe: {source_path}", "red")
        return

    # 1. Inicializar BagProcessor o Streamer
    if not is_dir:
        cprint(f"📂 Leyendo bag: {source_path}", "cyan")
        
        # --- NUEVO: Selección dinámica de Topics ---
        topics_dict = {
            "info_l": "/stereo_ch3/left/camera_info",
            "info_r": "/stereo_ch3/right/camera_info"
        }
        
        # Comprobar si 'compressed' está en el nombre del archivo
        if "compressed" in os.path.basename(source_path).lower():
            cprint("🗜️ Modo comprimido detectado: Usando topics /compressed", "yellow")
            topics_dict["left"] = "/stereo_ch3/left/image_raw/compressed"
            topics_dict["right"] = "/stereo_ch3/right/image_raw/compressed"
        else:
            cprint("📷 Modo normal detectado: Usando topics raw", "yellow")
            topics_dict["left"] = "/stereo_ch3/left/image_raw"
            topics_dict["right"] = "/stereo_ch3/right/image_raw"
            
        bag_proc = BagProcessor(source_path, topics_dict)
        cprint("🔍 Buscando mensajes de calibración...", "yellow")
        camera_info = bag_proc.get_calibration()
        image_iterator = bag_proc.stream_stereo_pairs()
    else:
        cprint(f"📂 Modo Carpeta: Leyendo desde {source_path}", "cyan")
        cprint("❌ El modo carpeta requiere adaptar la calibración manualmente.", "red")
        return

    if camera_info['left'] is None or camera_info['right'] is None:
        cprint("❌ Error: No se encontró info de calibración. No se puede rectificar.", "red")
        return
        
    bag_proc.save_calibration_yaml(current_out_path)
    cprint("✅ Calibración guardada.", "green")

    # 2. Inicializar Módulos
    img_proc = ImageProcessor(info_l=camera_info['left'], info_r=camera_info['right'])
    fish_detector = FishDetector(args.model_path, conf_thr=CONF_THR)
    bagfile_fauna = Bagfile_fauna(current_out_path, gt)
    stereo = StereoVision(calibration_data=camera_info, config_path=args.stereo_config, scale=decimation,image_channels=image_channels)
    
    # 3. Bucle de Procesamiento
    cprint(f"🚀 Iniciando procesamiento y exportación a: {current_out_path}", "cyan")
    count = 0
    
    for timestamp, img_l_raw, img_r_raw in image_iterator:
        fname = f"frame_{count}"
        clean_name = str(count) # Limpiamos fname para evitar problemas de rutas
    
        img_proc.set_image_pair(img_l_raw, img_r_raw)
        img_proc.rectify()
        
        # if count == 0: 
        #     cprint("🛠️ Abriendo verificador de rectificación...", "yellow")
        #     img_proc.check_rectification_interactive()
        
        img_proc.downsample(decimation)
        img_l, img_r = img_proc.get_processed()
        
        os.makedirs(os.path.join(current_out_path,"original_images"), exist_ok=True)
        
        cv2.imwrite(os.path.join(current_out_path,"original_images", fname+"_left.png"), img_l)
        cv2.imwrite(os.path.join(current_out_path,"original_images", fname+"_right.png"), img_r)
               
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
                # base_debug_folder = current_out_path,
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
            
            # Guardar escena en subcarpeta específica del frame usando clean_name
            scene_pcd = stereo.extract_point_cloud(scene_points_3d, img_l, mask=valid_disp_mask)
            frame_dir = os.path.join(current_out_path, f"frame_{clean_name}")
            os.makedirs(frame_dir, exist_ok=True)
            scene_ply_name = os.path.join(frame_dir, f"{clean_name}_scene.ply")
            
            stereo.save_point_cloud(scene_pcd, save_path=scene_ply_name)
        
        else:
            cprint(f"No fish found in frame {count} :(, gonna process next image!","yellow")
        
        count += 1
        print(f"Procesado frame par: {count}", end='\r')
        
    bagfile_fauna.process_and_save_df()
    cprint(f"\n✅ Bagfile terminado. {count} frames procesados.", "green")


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description="Script para procesar bagfiles y extraer 3D de peces.")
    
    # Opciones de Entrada
    parser.add_argument("--bag_file", "-bg", type=str, default=None, help="Ruta a un único bagfile")
    parser.add_argument("--images_source_dir", type=str, default=None, help="Lee imágenes de esta carpeta")
    parser.add_argument("--bags_dir", "-bd", type=str, default=BAGS_DIR, help="Carpeta padre para procesar MULTIPLES bagfiles")
    
    # Opciones de Salida
    parser.add_argument("--out_path", "-out", type=str, default=OUT_PATH, help="Carpeta raíz donde se guardarán los resultados")
    
    # Configuración de Modelos y Algoritmos
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path al modelo de IA YOLO")
    parser.add_argument("--stereo_config", type=str, default="/home/slimbook/fish_sizing/config/stereo_config.yaml", help="Path al config del stereo yaml")
    parser.add_argument("--selected_pipeline", default=SELECTED_PIPELINE, help="Pipeline de procesamiento de imagenes")
    parser.add_argument("--decimation", type=float, default=0.5, help="Escala para reducir imágenes (0.5 = mitad)")
    
    # Lógica de Filtrado 3D
    parser.add_argument("--ignore_borders", action="store_true", default=True, help="Procesa peces que tocan el borde")
    parser.add_argument("--ignore_completeness", action="store_true", default=True, help="Procesa peces incompletos")
    parser.add_argument("--ignore_overlap", action="store_true", default=True, help="Procesa peces que se solapan")
    parser.add_argument("--overlap_margin", type=float, default=0.05, help="Margen Z en metros para evaluar solapamiento")
    
    args = parser.parse_args()
    
    # Convertir rutas generales para Docker
    args.out_path = transform_path2docker(args.out_path)
    args.model_path = transform_path2docker(args.model_path)
    args.stereo_config = transform_path2docker(args.stereo_config)
    
    # Recopilar la lista de tareas a ejecutar
    tasks = [] # Lista de tuplas: (ruta_origen, ruta_salida, es_carpeta)
    
    if args.bags_dir:
        bags_dir = transform_path2docker(args.bags_dir)
        cprint(f"🔍 Escaneando directorio: {bags_dir}", "cyan")
        
        # Buscar recursivamente todos los .bag
        for root, _, files in os.walk(bags_dir):
            for file in files:
                if file.endswith(".bag"):
                    bag_path = os.path.join(root, file)
                    
                    # 1. Extraer el sufijo del bag (ej: '0' de '..._0.bag')
                    bag_name_no_ext = os.path.splitext(file)[0]
                    bag_suffix = bag_name_no_ext.split('_')[-1] 
                    
                    # 2. Mantener la estructura de subcarpetas (ej: '2025_05_08/11_36_35')
                    rel_path = os.path.relpath(root, bags_dir)
                    
                    # 3. Construir la ruta final
                    if rel_path == ".":
                        # Si los bags están en la raíz de bags_dir
                        specific_out = os.path.join(args.out_path, bag_suffix)
                    else:
                        # Si están en subcarpetas (lo normal en tu caso)
                        specific_out = os.path.join(args.out_path, rel_path, bag_suffix)
                        
                    tasks.append((bag_path, specific_out, False))
                    
    elif args.images_source_dir:
        source = transform_path2docker(args.images_source_dir)
        tasks.append((source, args.out_path, True))
    else:
        source = transform_path2docker(args.bag_file)
        tasks.append((source, args.out_path, False))
        
    
    if not tasks:
        cprint("❌ No se encontraron bagfiles para procesar.", "red")
        sys.exit(1)

    cprint(f"📋 Se procesarán {len(tasks)} orígenes en total.", "yellow")
    
    # Ejecutar el bucle principal
    for i, (source_path, current_out_path, is_dir) in enumerate(tasks):
        cprint("\n" + "="*60, "magenta")
        cprint(f"🚀 TAREA {i+1}/{len(tasks)}: PROCESANDO ORIGEN", "magenta", attrs=['bold'])
        cprint(f"📄 Origen: {source_path}", "magenta")
        cprint(f"📁 Salida: {current_out_path}", "magenta")
        cprint("="*60 + "\n", "magenta")
        
        try:
            # 1. Ejecutar todo el proceso
            process_single_source(source_path, current_out_path, is_dir, args)
            
            # 2. Limpiar y organizar la carpeta _inferred
            move_inferred_images(current_out_path)
            
        except Exception as e:
            cprint(f"❌ Error CRÍTICO procesando {source_path}: {e}", "red")

if __name__ == "__main__":
    main()