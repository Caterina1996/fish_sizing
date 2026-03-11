import os
import cv2
import logging
import argparse
import pickle
import numpy as np
from termcolor import cprint
import yaml
import open3d as o3d
import time

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


DEFAULT_INPUT    = "/home/slimbook/bagfiles/peixos_morts_piscina_v3/2025_05_08/11_28_42/stereo_camera_images_2025-05-08-11-28-42_0.bag"
DEFAULT_OUTPUT   = "/home/slimbook/results_fish_sizing/2025_05_08/11_28_42/0/"
DEFAULT_STEREO   = "/home/slimbook/fish_sizing/config/stereo_config.yaml"
DEFAULT_PIPELINE = "/home/slimbook/fish_sizing/config/pipeline_params.yaml"


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Modular Fish Sizing Pipeline")
    
    parser.add_argument("--input", default=DEFAULT_INPUT, 
                        help=f"Bagfile o carpeta de imágenes (Default: {DEFAULT_INPUT})")
    
    parser.add_argument("--out_path", default=DEFAULT_OUTPUT, 
                        help=f"Carpeta de resultados (Default: {DEFAULT_OUTPUT})")
    
    parser.add_argument("--stereo_config", default=DEFAULT_STEREO, 
                        help=f"YAML de calibración (Default: {DEFAULT_STEREO})")
    
    parser.add_argument("--pipeline_config_path", default=DEFAULT_PIPELINE, 
                        help=f"Parámetros del pipeline (Default: {DEFAULT_PIPELINE})")
    
    # Switches de control modular (se mantienen igual, False por defecto)
    parser.add_argument("--skip_extraction",default=False,  action="store_true")
    parser.add_argument("--skip_processing",default=False, action="store_true")
    parser.add_argument("--skip_inference",default=False, action="store_true")
    parser.add_argument("--skip_pc",default=False, action="store_true")
    
    args = parser.parse_args()
    cprint("50*+","cyan")
    cprint("SAVING CONFIG","cyan")
    cprint("50*+","cyan")


    # # --- 1. TRANSFORMACIÓN DE RUTAS ---
    args.input      = transform_path2docker(args.input)
    args.out_path     = transform_path2docker(args.out_path)
    args.stereo_config = transform_path2docker(args.stereo_config)
    args.pipeline_config_path   = transform_path2docker(args.pipeline_config_path)

    # --- 2. CARGAR CONFIGURACIÓN ---
    if not os.path.exists(args.pipeline_config_path):
        raise FileNotFoundError(f"No se encuentra el config: {args.pipeline_config_path}")
        
    with open(args.pipeline_config_path, 'r') as f:
        p_cfg = yaml.safe_load(f)
    
    model_path       = transform_path2docker(p_cfg.get('model_path'))
    decimation       = p_cfg.get('decimation', 0.5)
    image_channels   = p_cfg.get('image_channels', 1)
    use_wls          = p_cfg.get('use_wls', True)
    Visualize_online = p_cfg.get('visualize', False)
    CONF_THR         = p_cfg.get('conf_thr', 0.5)
    selected_pipeline = p_cfg.get('pipeline_name', 'basic')
    gt = p_cfg.get('gt_ground_truth', None)
    
    quality_params = {
        'ignore_borders': p_cfg.get('ignore_borders', False),
        'ignore_completeness': p_cfg.get('ignore_completeness', False),
        'ignore_overlap': p_cfg.get('ignore_overlap', True),
        'overlap_margin': p_cfg.get('overlap_margin', 0.05),
        'ignore_aspect_ratio': p_cfg.get('ignore_aspect_ratio', False),
        'aspect_ratio_thr': p_cfg.get('aspect_ratio_thr', 2.0)
    }

    # Inicializar Logger y carpetas
    os.makedirs(args.out_path, exist_ok=True)
    # setup_logger(args.out_path)
    save_run_config(args)
    
    orig_images_dir = os.path.join(args.out_path, "original_images")
    os.makedirs(orig_images_dir, exist_ok=True)

    # --- 3. GESTIÓN DE ITERADOR Y CALIBRACIÓN ---
    camera_info = None
    if not args.skip_extraction:
        cprint_and_log(f"📥 Modo: Extracción directa desde Bagfile", "cyan")
        bag_proc = BagProcessor(args.input, TOPICS_DICT)
        bag_proc.save_calibration_yaml(args.out_path)
        camera_info = bag_proc.get_calibration()
        image_iterator = bag_proc.stream_stereo_pairs()
    else:
        cprint_and_log(f"📂 Modo: Streaming desde carpeta {orig_images_dir}", "yellow")
        camera_info = load_camera_info_from_yaml(args.out_path, scale=decimation)
        image_iterator = stream_stereo_from_folder(orig_images_dir)
        
    # --- 4. INICIALIZACIÓN DE CLASES ---
    fish_detector = FishDetector(model_path, conf_thr=CONF_THR) if not args.skip_inference else None
    bagfile_fauna = Bagfile_fauna(args.out_path, gt)
    stereo = StereoVision(calibration_data=camera_info, config_path=args.stereo_config, scale=decimation, image_channels=image_channels)
    
    img_proc = ImageProcessor(
        info_l=camera_info['left'], 
        info_r=camera_info['right']
    )
    
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
            rect_l, rect_r = img_proc.get_processed()
            cv2.imwrite(os.path.join(orig_images_dir, f"{fname}_left.png"), rect_l)
            cv2.imwrite(os.path.join(orig_images_dir, f"{fname}_right.png"), rect_r)
        # Ectraction already done
        else:        
            rect_l, rect_r = img_l_raw, img_r_raw
            img_proc.processed_left = rect_l
            img_proc.processed_right = rect_r
        
        # --- PROCESSING ---
        if not args.skip_processing:      
            proc_l, proc_r = img_proc.run_pipeline(
                PROCESSING_PIPELINES[selected_pipeline], 
                base_debug_folder = args.out_path,
                frame_id = fname,
                visualize = Visualize_online
            )

        # --- INFERENCE ---
        frame_scene = None
        any_fish = False

        if not args.skip_inference:
            any_fish, frame_scene = fish_detector.process_frame(
                rect_l, 
                frame_id=fname, 
                disparity_img=None, 
                save_debug=p_cfg.get('save_debug', False), 
                debug_path=args.out_path, 
                save_obj=False # Ya lo guardamos más alante y así está completo
            )
               
        else:
            pkl_path = os.path.join(args.out_path, fname, f"{fname}.pkl")
            
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
            
            frame_dir = os.path.join(args.out_path, fname)
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
                                                debug_path = args.out_path)
                    # np.save(disp_npy_path, disparity_map) # Guardar para no repetir matching-> JA quedara a la frame scene
                
                frame_scene.disparity_map = disparity_map
                scene_points_3d = stereo.reproject_to_3d(disparity_map)     
                frame_scene.scene_points_3d = scene_points_3d
                
                # Generar y guardar la nube (PLY)
                valid_disp_mask = (disparity_map > stereo.min_valid_disparity)
                scene_pcd = stereo.extract_point_cloud(scene_points_3d, rect_l, mask=valid_disp_mask)
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
                                   img_l=rect_l, 
                                   bagfile_fauna=bagfile_fauna, 
                                   stereo=stereo, 
                                   args=args,
                                   **quality_params)
            
            bagfile_fauna = fish_sizer.measure_fish()

            # Guardamos el PKL con los resultados de la medición
            with open(os.path.join(frame_dir, f"{fname}.pkl"), 'wb') as f:
                pickle.dump(frame_scene, f)

    # --- 6. CIERRE Y REPORTE ESTADÍSTICO ---
    move_inferred_images(args.out_path)
    bagfile_fauna.process_and_save_df()
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / count if count > 0 else 0
    
    # 1. Extraer los datos de la fauna recolectada
    # (Ajusta los nombres de las listas si tu Bagfile_fauna los llama distinto)
    peces_3d = getattr(bagfile_fauna, 'fish_list', [])
    peces_2d = getattr(bagfile_fauna, 'fish_list_2D', []) # O fish_2d_list
    total_detectados = len(peces_3d) + len(peces_2d)
    
    # 2. Auditar los motivos de descarte
    motivos = {"bordes": 0, "aspect_ratio": 0, "solapamiento": 0, "incompleto": 0}
    for f in peces_2d:
        if getattr(f, 'in_image_borders', False):
            motivos["bordes"] += 1
        elif getattr(f, 'aspect_ratio', 0) < p_cfg.get('aspect_ratio_thr', 2.0):
            motivos["aspect_ratio"] += 1
        elif getattr(f, 'does_overlap', False) and getattr(f, 'is_front_fish', False) == False:
            motivos["solapamiento"] += 1
        else:
            motivos["incompleto"] += 1

    # 3. Empaquetar estadísticas
    args.execution_stats = {
        "total_frames_processed": count,
        "total_processing_time_s": round(total_time, 2),
        "average_time_per_frame_s": round(avg_time, 2),
        "average_fps": round(1.0 / avg_time, 2) if avg_time > 0 else 0,
        "fish_detected_total": total_detectados,
        "fish_measured_3D": len(peces_3d),
        "fish_discarded_2D": len(peces_2d),
        "discard_reasons": motivos
    }
    
    # 4. Guardar configuración y log
    save_run_config(args)
    
    # 5. Imprimir resumen bonito en consola
    cprint_and_log("\n" + "="*50, "cyan", attrs=["bold"])
    cprint_and_log("📊 REPORTE DE EJECUCIÓN", "cyan", attrs=["bold"])
    cprint_and_log("="*50, "cyan", attrs=["bold"])
    cprint_and_log(f"🏁 Carpeta: {args.out_path}", "magenta")
    cprint_and_log(f"⏱️ Tiempo Total: {total_time:.2f}s ({avg_time:.2f}s/frame | {args.execution_stats['average_fps']} FPS)", "cyan")
    cprint_and_log(f"🐟 Peces Detectados (YOLO): {total_detectados}", "white")
    cprint_and_log(f"✅ Medidos en 3D:           {len(peces_3d)}", "green", attrs=["bold"])
    cprint_and_log(f"🚫 Descartados (Solo 2D):   {len(peces_2d)}", "red")
    if len(peces_2d) > 0:
        cprint_and_log(f"   ├─ Por tocar bordes:     {motivos['bordes']}", "yellow")
        cprint_and_log(f"   ├─ Por aspect ratio:     {motivos['aspect_ratio']}", "yellow")
        cprint_and_log(f"   ├─ Por solapamiento:     {motivos['solapamiento']}", "yellow")
        cprint_and_log(f"   └─ Por 3D incompleto:    {motivos['incompleto']}", "yellow")
    cprint_and_log("="*50 + "\n", "cyan", attrs=["bold"])

if __name__ == "__main__":
    main()