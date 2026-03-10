import os
import cv2
import logging
import argparse
import pickle
import numpy as np
from termcolor import cprint
import yaml
import open3d as o3d

# Imports de tu paquete
from fish_sizing.utils.config import TOPICS_DICT, PROCESSING_PIPELINES, transform_path2docker
from fish_sizing.utils.tools import (
    save_run_config, cprint_and_log, move_inferred_images, 
    stream_stereo_from_folder, setup_logger
)
from fish_sizing.stereo.stereo import StereoVision, load_camera_info_from_yaml
from fish_sizing.bag_tools.bag_processor import BagProcessor
from fish_sizing.img_processing.image_processor import ImageProcessor
from fish_sizing.detection.fish2D import Fish2D, FrameScene  
from fish_sizing.detection.fish_detector import FishDetector
from fish_sizing.measurement.fish3D import Fish3D
from fish_sizing.analysis.bagfile_fauna import Bagfile_fauna
from fish_sizing.measurement.fish_sizer import FishSizer

def main():
    parser = argparse.ArgumentParser(description="Modular Fish Sizing Pipeline")
    parser.add_argument("--input", required=True, help="Bagfile o carpeta de imágenes")
    parser.add_argument("--output", required=True, help="Carpeta de resultados")
    parser.add_argument("--stereo_config", required=True, help="YAML de calibración estéreo")
    parser.add_argument("--pipeline_config", required=True, help="YAML de parámetros del pipeline")
    
    # Switches de control modular
    parser.add_argument("--skip_extraction", action="store_true")
    parser.add_argument("--skip_processing", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--skip_pc", action="store_true")
    
    args = parser.parse_args()

    # --- 1. TRANSFORMACIÓN DE RUTAS ---
    input_path      = transform_path2docker(args.input)
    output_path     = transform_path2docker(args.output)
    stereo_cfg_path = transform_path2docker(args.stereo_config)
    pipe_cfg_path   = transform_path2docker(args.pipeline_config)

    # --- 2. CARGAR CONFIGURACIÓN ---
    if not os.path.exists(pipe_cfg_path):
        raise FileNotFoundError(f"No se encuentra el config: {pipe_cfg_path}")
        
    with open(pipe_cfg_path, 'r') as f:
        p_cfg = yaml.safe_load(f)
    
    model_path       = transform_path2docker(p_cfg.get('model_path'))
    decimation       = p_cfg.get('decimation', 0.5)
    image_channels   = p_cfg.get('image_channels', 1)
    use_wls          = p_cfg.get('use_wls', True)
    gt               = p_cfg.get('gt', True) 
    Visualize_online = p_cfg.get('visualize', False)
    CONF_THR         = p_cfg.get('conf_thr', 0.5)
    selected_pipeline = p_cfg.get('pipeline_name', 'basic')

    # Inicializar Logger y carpetas
    os.makedirs(output_path, exist_ok=True)
    setup_logger(output_path)
    
    orig_images_dir = os.path.join(output_path, "original_images")
    os.makedirs(orig_images_dir, exist_ok=True)

    # --- 3. GESTIÓN DE ITERADOR Y CALIBRACIÓN ---
    camera_info = None
    if not args.skip_extraction:
        cprint_and_log(f"📥 Modo: Extracción directa desde Bagfile", "cyan")
        bag_proc = BagProcessor(input_path, TOPICS_DICT)
        bag_proc.save_calibration_yaml(output_path)
        camera_info = bag_proc.get_calibration()
        image_iterator = bag_proc.stream_stereo_pairs()
    else:
        cprint_and_log(f"📂 Modo: Streaming desde carpeta {orig_images_dir}", "yellow")
        camera_info = load_camera_info_from_yaml(output_path, scale=decimation)
        image_iterator = stream_stereo_from_folder(orig_images_dir)
        
    # --- 4. INICIALIZACIÓN DE CLASES ---
    fish_detector = FishDetector(model_path, conf_thr=CONF_THR) if not args.skip_inference else None
    bagfile_fauna = Bagfile_fauna(output_path, gt)
    stereo = StereoVision(calibration_data=camera_info, config_path=stereo_cfg_path, scale=decimation, image_channels=image_channels)
    img_proc = ImageProcessor()
    
    # --- 5. BUCLE DE PROCESAMIENTO UNIFICADO ---
    count = 0
    for timestamp, img_l_raw, img_r_raw in image_iterator:
        fname = f"frame_{count}"
        count += 1
        
        # --- EXTRACTION ---
        if not args.skip_extraction:
            img_proc.set_image_pair(img_l_raw, img_r_raw)
            img_proc.rectify()
            img_proc.downsample(decimation)
            proc_l, proc_r = img_proc.get_processed()
            cv2.imwrite(os.path.join(orig_images_dir, f"{fname}_left.png"), proc_l)
            cv2.imwrite(os.path.join(orig_images_dir, f"{fname}_right.png"), proc_r)
        else:        
            proc_l, proc_r = img_l_raw, img_r_raw
            img_proc.processed_left = img_l_raw
            img_proc.processed_right = img_r_raw
        
        # --- PROCESSING ---
        if not args.skip_processing:      
            proc_l, proc_r = img_proc.run_pipeline(
                PROCESSING_PIPELINES[selected_pipeline], 
                base_debug_folder = output_path,
                frame_id = fname,
                visualize = Visualize_online
            )

        # --- INFERENCE ---
        frame_scene = None
        any_fish = False

        if not args.skip_inference:
            any_fish, frame_scene = fish_detector.process_frame(
                proc_l, 
                frame_id=fname, 
                disparity_img=None, 
                save_debug=p_cfg.get('save_debug', False), 
                debug_path=output_path, 
                save_obj=True
            )
               
        else:
            pkl_path = os.path.join(output_path, fname, f"{fname}.pkl")
            
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    frame_scene = pickle.load(f)
                any_fish = True
                cprint(f"📦 {fname}: Cargando detección previa.", "yellow")
            else:
                any_fish = False
        
                
        ######################################################
        #          STEREO AND MEASUREMENT
        ######################################################
        if any_fish and frame_scene:
            
            frame_dir = os.path.join(output_path, fname)
            os.makedirs(frame_dir, exist_ok=True)
            
            disp_npy_path = os.path.join(frame_dir, f"{fname}_disparity.npy")
            scene_ply_path = os.path.join(frame_dir, f"{fname}_scene.ply")

            # --- LÓGICA DE RECONSTRUCCIÓN 3D / CARGA ---
            if not args.skip_pc:
                # Si no saltamos PC, necesitamos la disparidad (Cargar o Calcular)
                if os.path.exists(disp_npy_path):
                    cprint(f"📂 {fname}: Cargando disparidad (.npy) existente...", "blue")
                    disparity_map = np.load(disp_npy_path)
                else:
                    cprint(f"🔄 {fname}: Calculando disparidad (Matching + WLS)...", "green")
                    strips = frame_scene.get_optimization_strips()
                    disparity_map = stereo.compute_disparity(
                                                frame_id = fname,
                                                img_l = proc_l, 
                                                img_r = proc_r, 
                                                strips = strips, 
                                                use_wls = use_wls, 
                                                debug = False, 
                                                debug_path = output_path)
                    np.save(disp_npy_path, disparity_map) # Guardar para no repetir matching
                
                frame_scene.disparity_map = disparity_map
                scene_points_3d = stereo.reproject_to_3d(disparity_map)     
                frame_scene.scene_points_3d = scene_points_3d
                
                # Generar y guardar la nube (PLY)
                valid_disp_mask = (disparity_map > stereo.min_valid_disparity)
                scene_pcd = stereo.extract_point_cloud(scene_points_3d, proc_l, mask=valid_disp_mask)
                stereo.save_point_cloud(scene_pcd, save_path=scene_ply_path)
            
            else:
                # MODO SKIP PC: Solo queremos medir. 
                if os.path.exists(scene_ply_path):
                    cprint(f"☁️ {fname}: Cargando nube de puntos (.ply) para medir...", "cyan")
                    
                    pcd = o3d.io.read_point_cloud(scene_ply_path)
                    # El FishSizer necesita los puntos en numpy
                    frame_scene.scene_points_3d = np.asarray(pcd.points) 
                    
                elif os.path.exists(disp_npy_path):
                    cprint(f"⚠️ {fname}: PC skip activo pero PLY no encontrado. Reproyectando npy...", "yellow")
                    disparity_map = np.load(disp_npy_path)
                    frame_scene.scene_points_3d = stereo.reproject_to_3d(disparity_map)
                else:
                    cprint_and_log(f"❌ Error: Sin datos 3D para {fname}", "red", level=logging.ERROR)
                    continue

            # --- MEDICIÓN (FishSizer) ---
            # En este punto, frame_scene.scene_points_3d ya tiene datos XYZ vengan de donde vengan
            fish_sizer = FishSizer(frame_scene=frame_scene, 
                                   img_l=proc_l, 
                                   bagfile_fauna=bagfile_fauna, 
                                   stereo=stereo, 
                                   args=args)
            
            bagfile_fauna = fish_sizer.measure_fish()

            # Guardamos el PKL con los resultados de la medición
            with open(os.path.join(frame_dir, f"{fname}.pkl"), 'wb') as f:
                pickle.dump(frame_scene, f)

    # --- 6. CIERRE ---
    move_inferred_images(output_path)
    bagfile_fauna.process_and_save_df()
    save_run_config(output_path, args, CONF_THR, gt, Visualize_online, use_wls, image_channels)
    cprint_and_log(f"🏁 Finalizado: {output_path}", "magenta", attrs=["bold"])

if __name__ == "__main__":
    main()