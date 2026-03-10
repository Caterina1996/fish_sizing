#!/usr/bin/env python3.9
import argparse
import os
import sys
import yaml
import glob
import pickle
import numpy as np
import open3d as o3d
import cv2  # <--- AÑADIDO: Importante para leer el PNG
from termcolor import cprint, colored
from natsort import natsorted

# Importamos solo lo estrictamente necesario para medir
from fish_sizing.analysis.bagfile_fauna import Bagfile_fauna
from fish_sizing.measurement.fish3D import Fish3D

PATH_MAPPINGS = {
    "/home/slimbook/bagfiles": "/home/rosuser/dataset/bagfiles",
    "/home/slimbook/fish_sizing/out": "/home/rosuser/repo/out",
    "/home/slimbook/models": "/home/rosuser/dataset/models/",
    "/home/slimbook/fish_sizing/config" :"/home/rosuser/repo/config/",
    "/media/slimbook/easystore": "/home/rosuser/easystore"
}

USE_DOCKER = True

def transform_path2docker(path: str) -> str:
    if not USE_DOCKER or path is None:
        return path
    for host_path, docker_path in PATH_MAPPINGS.items():
        if host_path in path:
            new_path = path.replace(host_path, docker_path)
            cprint(f"🔄 Path mapped: {path} \n   -> {new_path}", "yellow")
            return new_path 
    return path


def check_3d_offline(fish, frame_scene, focal, baseline, args):
    """
    Versión offline del check_3d. 
    """
    x1, y1, x2, y2 = fish.bbox
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
    
    is_front_fish = True 
    if fish.does_overlap and frame_scene.disparity_map is not None:
        my_disp_values = frame_scene.disparity_map[fish.mask > 0]
        my_median_disp = np.median(my_disp_values) if len(my_disp_values) > 0 else 0

        for neighbour_id in fish.overlapping_ids:
            neighbour = next((f for f in frame_scene.fish_list if f.track_id == neighbour_id), None)
            if neighbour and neighbour.mask is not None:
                neighbour_disp_values = frame_scene.disparity_map[neighbour.mask > 0]
                neigh_median_disp = np.median(neighbour_disp_values) if len(neighbour_disp_values) > 0 else 0

                if my_median_disp > 0.1 and neigh_median_disp > 0.1:
                    my_depth = (focal * baseline) / my_median_disp
                    neigh_depth = (focal * baseline) / neigh_median_disp
                    
                    if neigh_depth < (my_depth + args.overlap_margin):
                        is_front_fish = False
                        break 

    fish.aspect_ratio = aspect_ratio
    fish.is_front_fish = is_front_fish 

    cond_aspect_ratio  = (aspect_ratio >= args.aspect_ratio_thr) or args.ignore_aspect_ratio
    cond_completeness  = fish.is_3d_complete or args.ignore_completeness
    cond_borders       = not fish.in_image_borders or args.ignore_borders
    cond_overlap_smart = (not fish.does_overlap) or is_front_fish or args.ignore_overlap
    
    fish_3d_ok = cond_aspect_ratio and cond_completeness and cond_borders and cond_overlap_smart
    fish.fish_3d_ok = fish_3d_ok
    
    print(colored(f"\n--- ESTADO FÍSICO FISH {fish.track_id} ---", "cyan"))
    print(f"Aspect Ratio: {aspect_ratio:.2f} | Completo: {fish.is_3d_complete} | Toca Borde: {fish.in_image_borders}")
    print(f"Solapado: {fish.does_overlap} | Está Delante: {fish.is_front_fish}")
    
    if not fish_3d_ok:
        if not cond_aspect_ratio:
            cprint(f"🚫 Descartado: De cara o curvado (Ratio {aspect_ratio:.2f} < {args.aspect_ratio_thr})", "magenta")
        else:
            cprint(f"🚫 Descartado para medir 3D según tus reglas de ignorar", "magenta")
    else:
        cprint(f"✅ Aprobado para cálculo 3D", "green")
        
    return fish_3d_ok


def main():
    parser = argparse.ArgumentParser(description="Reprocesa medidas 3D desde caché (.pkl y .ply) sin recalcular estéreo.")
    parser.add_argument("--in_dir", "-in", type=str, required=True, help="Carpeta de resultados anterior")
    parser.add_argument("--out_dir", type=str, default=None, help="Carpeta destino. Si no se pone, se usa in_dir/to_paper")
    parser.add_argument("--gt", type=float, default=28.9, help="Ground Truth en cm para esta carpeta")
    
    # Flags de filtrado
    parser.add_argument("--ignore_borders", action="store_true", default=False)
    parser.add_argument("--ignore_completeness", action="store_true", default=False)
    parser.add_argument("--ignore_overlap", action="store_true", default=True)
    parser.add_argument("--overlap_margin", type=float, default=0.05)
    parser.add_argument("--ignore_aspect_ratio", action="store_true", default=False)
    parser.add_argument("--aspect_ratio_thr", type=float, default=1.8)
    
    args = parser.parse_args()
    args.in_dir = transform_path2docker(args.in_dir)
    
    if args.out_dir is None:
        args.out_dir = os.path.join(args.in_dir, "to_paper")
    else:
        args.out_dir = transform_path2docker(args.out_dir)
    
    if not os.path.exists(args.in_dir):
        cprint(f"❌ Error: La carpeta {args.in_dir} no existe.", "red")
        sys.exit(1)

    # 1. AUTO-CALIBRACIÓN LIGERA
    # focal = 1460.0 
    # baseline = 0.12
    
    right_yaml = os.path.join(args.in_dir, "right.yaml")
    if os.path.exists(right_yaml):
        with open(right_yaml, 'r') as f:
            cam_data = yaml.safe_load(f)
            P = cam_data.get('projection_matrix', {}).get('data', [])
            if len(P) > 3:
                focal = P[0]
                baseline = abs(P[3] / P[0]) if P[0] != 0 else 0.12
                cprint(f"📐 Calibración cargada: F={focal:.1f}, B={baseline:.3f}", "green")

    bagfile_fauna = Bagfile_fauna(args.out_dir, gt=args.gt)
    
    # 3. BUSCAR TODOS LOS FRAMES CACHEADOS (.pkl) EN IN_DIR
    pkl_files = natsorted(glob.glob(os.path.join(args.in_dir, "*.pkl")))
    
    if not pkl_files:
        cprint(f"⚠️ No se encontraron archivos .pkl de YOLO en {args.in_dir}", "yellow")
        sys.exit(0)

    cprint(f"🚀 Iniciando Reprocesamiento en Caché ({len(pkl_files)} frames detectados)", "cyan")
    cprint(f"📁 Guardando nuevos resultados en: {args.out_dir}/results", "cyan")

    # 4. BUCLE PRINCIPAL
    for pkl_path in pkl_files:
        
        with open(pkl_path, 'rb') as f:
            frame_scene = pickle.load(f)
            
        frame_name = str(frame_scene.frame_name)
        frame_dir = os.path.join(args.in_dir, frame_name) 
        
        if not os.path.exists(frame_dir):
            continue

        disp_npy_path = os.path.join(frame_dir, f"{frame_name}_disparity.npy")
        disp_png_path = os.path.join(frame_dir, f"{frame_name}_disparity.png")
        
        if os.path.exists(disp_npy_path):
            frame_scene.disparity_map = np.load(disp_npy_path)
        elif os.path.exists(disp_png_path):
            # Leemos en blanco y negro y convertimos colores a matriz 3D válida
            disp_img = cv2.imread(disp_png_path, cv2.IMREAD_GRAYSCALE)
            frame_scene.disparity_map = np.where(disp_img > 5, 1000.0, 0.0).astype(np.float32)
        else:
            frame_scene.disparity_map = None
        # ------------------------------------------------------------

        # C) Iterar peces
        for fish in frame_scene.fish_list:
            if fish.mask is None: 
                continue
            
            if frame_scene.disparity_map is not None:
                fish.is_complete(frame_scene.disparity_map, debug_path=args.out_dir, debug_mode=False)

            fish_ply_path = os.path.join(frame_dir, f"{frame_name}_{fish.color_id}.ply")
            
            if not os.path.exists(fish_ply_path):
                cprint(f"⚠️ Nube 3D no encontrada para Fish {fish.track_id} en {frame_name}", "yellow")
                bagfile_fauna.add_fish_2D(fish)
                continue

            pcd = o3d.io.read_point_cloud(fish_ply_path)
            if pcd.is_empty():
                bagfile_fauna.add_fish_2D(fish)
                continue

            points_np = np.asarray(pcd.points)
            colors_bgr_int = (np.asarray(pcd.colors)[:, ::-1] * 255).astype(np.uint8)

            current_fish_3d = Fish3D(fish, points_np, args.out_dir, colors_bgr_int)

            fish_3d_ok = check_3d_offline(fish, frame_scene, focal, baseline, args)
            
            current_fish_3d.fish_3d_ok = fish_3d_ok
            current_fish_3d.aspect_ratio = getattr(fish, 'aspect_ratio', 0.0)
            current_fish_3d.is_front_fish = getattr(fish, 'is_front_fish', True)
            
            current_fish_3d.is_3d_complete = getattr(fish, 'is_3d_complete', False)
            current_fish_3d.in_image_borders = getattr(fish, 'in_image_borders', False)
            current_fish_3d.does_overlap = getattr(fish, 'does_overlap', False)
            current_fish_3d.overlapping_ids = getattr(fish, 'overlapping_ids', [])

            # 2. CORTAFUEGOS (Early Exit)
            dealbreaker_border = fish.in_image_borders and not args.ignore_borders
            dealbreaker_completeness = not fish.is_3d_complete and not args.ignore_completeness
            
            if dealbreaker_border or dealbreaker_completeness:
                cprint(f"   ⏭️ Saltando CPU para Fish {fish.track_id} (Bordes/Incompleto)", "yellow")
                bagfile_fauna.add_fish(current_fish_3d)
                continue

            cprint(f" 🐟 ⚙️ Midiendo Fish {fish.track_id}...", "yellow")
            
            current_fish_3d.filter_outliers_HDBSCAN_adaptive(debug_plot=False)
            current_fish_3d.get_distance_camera_fish()
            filt_saved = current_fish_3d.save_fish_pointcloud(filtered=True)
            
            if filt_saved:
                current_fish_3d.measure_fish_length_ply_with_angles_and_plot(plot_fish_direction=False, filtered=True)
                spine_len = current_fish_3d.measure_curved_length(filtered=True, num_slices=5)
                current_fish_3d.spine_length = spine_len
                
            bagfile_fauna.add_fish(current_fish_3d)

    cprint(f"\n✅ Procesamiento terminado. Generando CSVs...", "green")
    bagfile_fauna.process_and_save_df()

if __name__ == "__main__":
    main()