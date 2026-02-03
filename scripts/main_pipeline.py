#!/usr/bin/env python3.9
import argparse
import os
import sys
import cv2
import numpy as np
from termcolor import cprint

from fish_sizing.utils.bag_processor import BagProcessor
from fish_sizing.utils.image_processor import ImageProcessor
from fish_sizing.detection.fish2D import Fish2D, FrameScene  # pot ser aquest import ens el podem estalviar
from fish_sizing.stereo.stereo import StereoVision
from fish_sizing.detection.fish_detector import  FishDetector
from fish_sizing.measurement.fish3D import  Fish3D
from fish_sizing.analysis.bagfile_fauna import  Bagfile_fauna


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

BAGFILE_PATH="/home/slimbook/bagfiles/peixos_morts_piscina_v3/2025_05_08/11_17_18/stereo_camera_images_2025-05-08-11-18-25_1.bag"
MODEL_PATH = "/home/slimbook/models/yv11l/ylarge_d18_poolv2r_lantytr_nocturnes/weights/best.pt"
OUT_PATH = "/home/slimbook/fish_sizing/out/test_export/2025-05-08-11-18-25_1/"

# --- CONFIGURACIÓN DE PIPELINES ---
PROCESSING_PIPELINES = {
    "basic": [
        # 1. Igualar luz (No guardamos, es un paso intermedio sutil)
        ("match_histograms", {"reference": "left"}, True),
        
        # 2. Convertir a gris pasando del rojo
        ("convert_to_custom_grayscale", {}, True),
        
        # 3. CLAHE (Guardamos, paso crítico que cambia mucho la imagen)
        ("apply_clahe", {"clip_limit": 2.0, "grid_size": (8,8)}, True)
        
    ],

    "experiment_dehaze": [
        # Aquí queremos ver qué tal funciona el dehaze, así que True
        ("apply_dehaze",            {"omega": 0.85}, True), 
        ("match_brightness_linear", {"reference": "left"}, False),
        ("apply_clahe",             {"clip_limit": 2.0}, True)
    ]
}

SELECTED_PIPELINE = "basic"

# img_proc.apply_dehaze(omega=0.85, window_size=15,stereo_consistency=True) #-> Revisar esto xq ahora mismo no va be/no interesa
# # img_proc.visualize_and_save()
# img_l, img_r = img_proc.get_processed()

# img_proc.match_histograms(reference="left") # Igualar brillos
# # img_proc.visualize_and_save()
# img_proc.convert_to_custom_grayscale()
# # img_proc.visualize_and_save()
# img_proc.apply_clahe()
# # img_proc.visualize_and_save()
# img_proc.match_histograms(reference="left")
# # img_proc.visualize_and_save()
# processed_l, processed_r = img_proc.get_processed()


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



# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Script para procesar bagfiles y guardar imágenes filtradas.")
    
    parser.add_argument("--bag_file", "-bg", type=str, help="Ruta al bagfile",
                        default=BAGFILE_PATH)

    parser.add_argument("--out_path", "-out", type=str, help="Carpeta donde se guardarán las imágenes procesadas",
                        default=OUT_PATH)
    
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
    
    parser.add_argument("--selected_pipeline", default=SELECTED_PIPELINE,
                        help="Pipeline de procesamiento de imagenes")
       

    args = parser.parse_args()
    
    # 1. Transformar rutas para Docker
    bag_file = transform_path2docker(args.bag_file)
    out_path = transform_path2docker(args.out_path)
    model_path = transform_path2docker(args.model_path)
    stereo_config_path = transform_path2docker(args.stereo_config)
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
    
    fish_detector = FishDetector(model_path)
    bagfile_fauna = Bagfile_fauna(out_path)
    
    stereo = StereoVision(calibration_data=camera_info, config_path=stereo_config_path,scale=decimation)
    
    # 5. Bucle de Procesamiento (Stereo Stream)
    cprint(f"🚀 Iniciando procesamiento y exportación a: {out_path}", "cyan")
    
    count = 0
    # Usamos stream_stereo_pairs porque necesitamos AMBAS imágenes para rectificar
    for timestamp, img_l_raw, img_r_raw in bag_proc.stream_stereo_pairs():
    
        if count > 20:
            break
        
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
            
            cv2.imwrite(os.path.join(out_path, fname+"original_left.png"), processed_l)
            cv2.imwrite(os.path.join(out_path, fname+"original_right.png"), processed_r)
            
            processed_l, processed_r = img_proc.run_pipeline(
                PROCESSING_PIPELINES[args.selected_pipeline], 
                base_debug_folder = out_path,
                frame_id = fname
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
                                    debug=True, 
                                    debug_path = out_path)
    
            # Inyectar disparidad en la escena y validar peces
            frame_scene.disparity_image = disparity_map
            
            scene_ply_name = os.path.join(out_path, f"{fname}_scene.ply")
            all_fish_ply_name = os.path.join(out_path, f"{fname}_all_fish.ply")
            
            # 2.1 Reproyectar toda la imagen a XYZ
            scene_points_3d = stereo.reproject_to_3d(disparity_map)
            
            # Inicializar la máscara para seleccionar el trozo de pc que queremos
            all_pc_mask = np.zeros(disparity_map.shape, dtype=bool)
            all_fish_mask = np.zeros(disparity_map.shape, dtype=bool)
            
            # Filtrar las distancias > dist__max (no me fio de mesures més enfora de 4m)
            valid_disp_mask = (disparity_map > stereo.min_valid_disparity)
            
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 3. Measure and log every detected fish
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            for fish in frame_scene.fish_list:
                if fish.mask is not None:
                    
                    # Log fish info
                    fish_class = fish.class_name  
                    track_id = fish.track_id  
                    fish.is_complete(disparity_map, debug_path = os.path.join(out_path,"debug"), debug_mode=True)
                    
                    cprint("FISH SUMMARY","cyan")
                    print("object number: ",fish.color_id)
                    print("fish class: ",fish_class)
                    print("Is fish complete? ",fish.is_3d_complete)
                    print("IS fish in the borders of the image?",fish.in_image_borders)
                    print("Does the fish overlap with others",fish.does_overlap)
                    print("With whom?",fish.overlapping_ids)
                    cprint("+++++++++++++++++++++++++++++++++++++++++","cyan")
                    
                    # Save fish pc
                    fish_ply_path = os.path.join(out_path, f"{fname}_{fish.color_id}.ply")
                    fish_pcd = stereo.extract_point_cloud(scene_points_3d, img_l, mask=fish.mask)
                    stereo.save_point_cloud(fish_pcd,save_path=fish_ply_path)     
                    
                    # Convert to numpy array to fish3D class 
                    # TODO! -> check if we can skip this transformation here
                    # Pot ser això podria estar a la classe fish3D o millor aquí?
                    points_np = np.asarray(fish_pcd.points) # (N, 3) float64
                    colors_rgb_float = np.asarray(fish_pcd.colors)
                    colors_bgr_int = (colors_rgb_float[:, ::-1] * 255).astype(np.uint8)
                        
                    current_fish_3d = Fish3D(fish, points_np, out_path,colors_bgr_int)
                    
                    # Decide wether to measure this fish or not
                    cond_completeness = fish.is_3d_complete or args.ignore_completeness
                    cond_borders = not fish.in_image_borders or args.ignore_borders
                    cond_overlap = not fish.does_overlap or args.ignore_overlap             
                    fish_3d_ok = cond_completeness and cond_borders and cond_overlap

                    if fish_3d_ok:
                        
                        cprint(f" 🐟 ⚙️ Procesando Fish {track_id}...", "yellow")
                        # A) Filtrado Avanzado (HDBSCAN adaptativo)
                        current_fish_3d.filter_outliers_HDBSCAN_adaptive()
                        current_fish_3d.get_distance_camera_fish()

                        raw_saved = current_fish_3d.save_fish_pointcloud(filtered=False)
                        filt_saved = current_fish_3d.save_fish_pointcloud(filtered=True)
                        
                        # C) MEDICIÓN (PCA + PyntCloud)
                        if raw_saved:
                            print(f"   📏 Midiendo Raw...")
                            current_fish_3d.measure_fish_length_ply_with_angles_and_plot(
                                plot_fish_direction=True, filtered=False)
                        
                        if filt_saved:
                            print(f"   📏 Midiendo Filtrado...")
                            current_fish_3d.measure_fish_length_ply_with_angles_and_plot(
                                plot_fish_direction=True, filtered=True)
                            
                        bagfile_fauna.add_fish(current_fish_3d)

                    else:
                        cprint(f"   ⚠️ Saltando medición Fish {track_id} (Incompleto o en borde)", "red")
                        print(f"⚠️ path to scene {frame_scene_path}: sin pointcloud — se añadirá con valores None")
                        print("SCENE:",scene_object.fish_list)
                        for fish_2d in scene_object.fish_list:
                            print("fish yey!! fshhhhhhhhhhhh")
                            bagfile_fauna.add_fish_2D(fish_2d)

                    # Acumular máscara para guardar "all_fish" después
                    all_fish_mask = all_fish_mask | (fish.mask > 0)
                    
            # Save all fish combined
            all_fish_pcd = stereo.extract_point_cloud(scene_points_3d, img_l, mask=all_fish_mask)
            stereo.save_point_cloud(all_fish_pcd,save_path=all_fish_ply_name)
            
            # Save scene
            scene_pcd = stereo.extract_point_cloud(scene_points_3d, img_l, mask=valid_disp_mask)
            stereo.save_point_cloud(scene_pcd,save_path=scene_ply_name)
        
        else:
            cprint(f"No fish found in frame {count} :(, gonna process next image!","yellow")
        
        count += 1
        print(f"Procesado frame par: {count}", end='\r')
    
    bagfile_fauna.process_and_save_df()

    cprint(f"\n✅ Terminado. {count} pares guardados en {out_path}", "green")

if __name__ == "__main__":
    main()

