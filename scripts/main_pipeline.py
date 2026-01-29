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

# # 1. Definir la ruta "mala" de ROS
# ros_path = '/opt/ros/noetic/lib/python3/dist-packages'

# # 2. Si está en el path, la quitamos temporalmente
# if ros_path in sys.path:
#     sys.path.remove(ros_path)

# # 3. AHORA importamos cv2 (Cogerá el de tu usuario ~/.local/...)
# import cv2

# # 4. (Opcional) Volvemos a meter la ruta de ROS por si necesitas 'rospy' luego
# sys.path.append(ros_path)

# # --- Resto de tus imports ---
# import numpy as np

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
    
    parser.add_argument("--bag_file", "-bg", type=str, 
                        default="/home/slimbook/bagfiles/LIMA/2025/2025_08_21/selec2/13_05_55/stereo_camera_images_2025-08-21-13-05-55_0.bag",
                        help="Ruta al bagfile")
    
    parser.add_argument("--out_path", "-out", type=str, 
                        default="/home/slimbook/fish_sizing/out/test_export",
                        help="Carpeta donde se guardarán las imágenes procesadas")
    
    parser.add_argument("--topic", type=str, default="left", choices=["left", "right"],
                        help="Qué cámara exportar")
    
    parser.add_argument("--model_path", type=str, default="/home/slimbook/models/yv11l/ylarge_d18_poolv2r_lantytr_nocturnes/weights/best.pt", 
                        help="Path to the detection AI model")
                        
    parser.add_argument("--stereo_config", type=str, default="/home/slimbook/fish_sizing/config/stereo_config.yaml", 
                        help="Path to the yaml with the config for the stereo matching alg")
    
    parser.add_argument("--decimation", type=float, default=0.5, 
                        help="reescale the images")
    
    args = parser.parse_args()
    
    # 1. Transformar rutas para Docker
    bag_file = transform_path2docker(args.bag_file)
    out_path = transform_path2docker(args.out_path)
    model_path = transform_path2docker(args.model_path)
    stereo_config_path = transform_path2docker(args.stereo_config)
    decimation = args.decimation

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
    
    stereo = StereoVision(calibration_data=camera_info, config_path=stereo_config_path,scale=decimation)
    
    # 5. Bucle de Procesamiento (Stereo Stream)
    cprint(f"🚀 Iniciando procesamiento y exportación a: {out_path}", "cyan")
    
    count = 0
    # Usamos stream_stereo_pairs porque necesitamos AMBAS imágenes para rectificar
    for timestamp, img_l_raw, img_r_raw in bag_proc.stream_stereo_pairs():
    
        if count > 2:
            break
        
        # frame name TODO: decidir si vull el timestamp o count per facilitat
        fname = f"{timestamp}"
        
        
        # Load the stereo pair
        img_proc.set_image_pair(img_l_raw, img_r_raw)
        
        # 1. Rectificar and downsample (we will use decimated_x2 images)
        img_proc.rectify()
        
        if count == 0: # Sanity check
            cprint("🛠️ Abriendo verificador de rectificación...", "yellow")
            img_proc.check_rectification_interactive()
        
        img_proc.downsample(decimation)
               
        # 2. Look for fish in the scene
        any_fish, frame_scene = fish_detector.process_frame(img_proc.processed_left, 
                                    frame_id=count, # decide weather to use this or timestamp 
                                    disparity_img=None, 
                                    save_debug=True, 
                                    debug_path=out_path, 
                                    save_obj=True)
        if any_fish:
            
            # 1. Process image to improve the stereo matching later then 
            img_proc.apply_dehaze(omega=0.85, window_size=15,stereo_consistency=True) #-> Revisar esto xq ahora mismo no va be/no interesa
            img_proc.match_histograms(reference="left") # Igualar brillos
            img_proc.apply_clahe()
            processed_l, processed_r = img_proc.get_processed()
            
            # Guardar processed
            # fname = f"{timestamp}"

            cv2.imwrite(os.path.join(out_path, fname+"_left.png"), processed_l)
            cv2.imwrite(os.path.join(out_path, fname+"_right.png"), processed_r)
            
            # get strips for the calculation of the disparity
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
            
            # E) REPROYECCIÓN 3D (Point Cloud)
            # -----------------------------------------------------------
            # 1. Reproyectar toda la imagen a XYZ
            points_3d = stereo.reproject_to_3d(disparity_map)
            
            # 2. Crear máscara combinada de todos los peces
            # (Iteramos sobre los peces detectados para sumar sus máscaras)
            combined_mask = np.zeros(disparity_map.shape, dtype=bool)
            
            # Añadimos píxeles con disparidad válida (filtro físico)
            valid_disp_mask = (disparity_map > stereo.min_valid_disparity)
            
            # Si YOLO devolvió máscaras, las usamos para filtrar solo el pez
            if frame_scene.fish_list:
                for fish in frame_scene.fish_list:
                    if fish.mask is not None:
                        # Asegurar que es binaria y sumar
                        combined_mask = combined_mask | (fish.mask > 0)
                
                # Máscara Final = (Donde hay pez) AND (Donde hay disparidad válida)
                final_mask = combined_mask & valid_disp_mask
            else:
                # Si no hay máscaras (solo cajas), usamos toda la disparidad válida
                final_mask = valid_disp_mask

            # 3. Guardar Nube de Puntos (PLY)
            ply_filename = os.path.join(out_path, f"{fname}_scene.ply")
            stereo.save_point_cloud(
                points_3d=points_3d, 
                colors=processed_l, 
                mask=final_mask, 
                save_path=ply_filename, 
                z_max=stereo.MAX_DEPTH_METERS
            )

        
        else:
            cprint(f"No fish found in frame {count} :(, gonna process next image!","yellow")
        
        count += 1
        print(f"Procesado frame par: {count}", end='\r')

    cprint(f"\n✅ Terminado. {count} pares guardados en {out_path}", "green")

if __name__ == "__main__":
    main()