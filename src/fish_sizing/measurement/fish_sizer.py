import numpy as np
import os
from termcolor import colored

import open3d as o3d
from sklearn.decomposition import PCA
from scipy import stats
from pyntcloud import PyntCloud
import cloudpickle as pickle
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import hdbscan
import collections
from termcolor import cprint

from fish_sizing.utils import tools
from fish_sizing.detection.fish2D import Fish2D, FrameScene
from fish_sizing.measurement.fish3D import  Fish3D

class FishSizer():
    def __init__(self, frame_scene, img_l,bagfile_fauna, stereo, args):
        
        self.frame_scene = frame_scene
        self.bagfile_fauna = bagfile_fauna
        self.img_l = img_l
        self.stereo = stereo
        
         
        if args is not None:
            self.__dict__.update(vars(args))
        
        # --- NUEVO: Definir y crear la carpeta específica del frame ---
        self.frame_dir = os.path.join(self.out_path, f"{self.frame_scene.frame_name}")
        os.makedirs(self.frame_dir, exist_ok=True)
        
        self.scene_ply_name = os.path.join(self.frame_dir, f"{self.frame_scene.frame_name}_scene.ply")
        self.all_fish_ply_name = os.path.join(self.frame_dir, f"{self.frame_scene.frame_name}_all_fish.ply")
            
    
    @staticmethod
    def _print_fish_summary(fish):
        print(colored("--- FISH SUMMARY ---", "cyan"))
        print(f"ID: {fish.track_id} | Class: {fish.class_name}")
        print(f"Complete: {fish.is_3d_complete} | Borders: {fish.in_image_borders}")
        print(f"Overlap: {fish.does_overlap} {fish.overlapping_ids if fish.does_overlap else ''}")
        print(colored("--------------------", "cyan"))
            
    def measure_fish(self,save_scene=True):

        """ Measure and log every detected fish"""
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
        # Inicializar la máscara para seleccionar el trozo de pc que queremos

        all_fish_mask = np.zeros(self.frame_scene.disparity_map.shape, dtype=bool)
        
        for fish in self.frame_scene.fish_list:
            if fish.mask is not None:
                # Log fish info
                fish_class = fish.class_name  
                track_id = fish.track_id  
                
                fish.is_complete(self.frame_scene.disparity_map, debug_path = os.path.join(self.out_path,"debug"), debug_mode=True)
                
                FishSizer._print_fish_summary(fish)
                
                fish_ply_path = os.path.join(self.frame_dir, f"{self.frame_scene.frame_name}_{fish.color_id}.ply")
                fish_pcd = self.stereo.extract_point_cloud(self.frame_scene.scene_points_3d, self.img_l, mask=fish.mask)
                
                if fish_pcd is None:
                    cprint(f"   ⚠️ Nube vacía para Fish {fish.track_id}. Saltando...", "red")
                    self.bagfile_fauna.add_fish_2D(fish)
                    continue
                
                self.stereo.save_point_cloud(fish_pcd,save_path=fish_ply_path)
                
                # Convert to numpy array to fish3D class 
                points_np = np.asarray(fish_pcd.points) # (N, 3) float64
                colors_rgb_float = np.asarray(fish_pcd.colors)
                colors_bgr_int = (colors_rgb_float[:, ::-1] * 255).astype(np.uint8)
                    
                current_fish_3d = Fish3D(fish, points_np, self.out_path,colors_bgr_int)
                
                # Decide if the fish is worth measuring (not in_image_borders, not overlapping and disparity ok)      
                fish_3d_ok =  self.check_3d(fish)        
                if fish_3d_ok:
                    cprint(f" 🐟 ⚙️ Procesando Fish {track_id}...", "yellow")
                    
                    # Save fish pc
                   
                    self.stereo.save_point_cloud(fish_pcd,save_path=fish_ply_path)  
                    
                    # A) Filtrado y Medición 3D
                    current_fish_3d.filter_outliers_HDBSCAN_adaptive(debug_plot=False)
                    # current_fish_3d.filter_outliers_HDBSCAN_adaptive() -> NAH
 
                    current_fish_3d.get_distance_camera_fish()

                    raw_saved = current_fish_3d.save_fish_pointcloud(filtered=False)
                    filt_saved = current_fish_3d.save_fish_pointcloud(filtered=True)
                    
                    if raw_saved:
                        print(f"   📏 Midiendo Raw...")
                        current_fish_3d.measure_fish_length_ply_with_angles_and_plot(
                            plot_fish_direction=True, filtered=False)
                    
                    if filt_saved:
                        print(f"   📏 Midiendo Filtrado...")
                        current_fish_3d.measure_fish_length_ply_with_angles_and_plot(
                            plot_fish_direction=True, filtered=True)
                        
                        # Medimos también usando el spine
                        spine_len = current_fish_3d.measure_curved_length(filtered=True, num_slices=5)
                        current_fish_3d.spine_length = spine_len # Guardamos en el objeto
                        
                        if spine_len > 0:
                            print(colored(f"   🐍 Longitud Curva (Spine): {spine_len * 100:.2f} cm", "green"))
                        
                    # Añadir como pez 3D válido
                    self.bagfile_fauna.add_fish(current_fish_3d)

                else:
                    # CASO FALLIDO (Incompleto, borde o solapado)
                    # Solo añadimos ESTE pez como entrada 2D (con valores None en 3D)
                    cprint(f"   ⚠️ Saltando medición 3D Fish {track_id} (No cumple condiciones)", "red")
                    self.bagfile_fauna.add_fish_2D(fish)

                # Acumular máscara para guardar "all_fish" después
                all_fish_mask = all_fish_mask | (fish.mask > 0)
                
        if save_scene:
            # Save all fish combined
            all_fish_pcd = self.stereo.extract_point_cloud(self.frame_scene.scene_points_3d, self.img_l, mask=all_fish_mask)
            self.stereo.save_point_cloud(all_fish_pcd,save_path=self.all_fish_ply_name)
            
        return self.bagfile_fauna
    
    def check_3d(self, fish):  
            
            # ==========================================
            # 1. EVALUAR REALIDAD FÍSICA Y GEOMÉTRICA
            # ==========================================
            
            # A) Aspect Ratio (Perfil vs De Cara)
            x1, y1, x2, y2 = fish.bbox
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
            
            # B) Solapamiento y Profundidad
            is_front_fish = True 
            if fish.does_overlap:
                my_disp_values = self.frame_scene.disparity_map[fish.mask > 0]
                my_median_disp = np.median(my_disp_values) if len(my_disp_values) > 0 else 0

                for neighbour_id in fish.overlapping_ids:
                    neighbour = next((f for f in self.frame_scene.fish_list if f.track_id == neighbour_id), None)
                    if neighbour and neighbour.mask is not None:
                        neighbour_disp_values = self.frame_scene.disparity_map[neighbour.mask > 0]
                        neigh_median_disp = np.median(neighbour_disp_values) if len(neighbour_disp_values) > 0 else 0

                        if my_median_disp > 0.1 and neigh_median_disp > 0.1:
                            my_depth = (self.stereo.FOCAL * self.stereo.BASELINE) / my_median_disp
                            neigh_depth = (self.stereo.FOCAL * self.stereo.BASELINE) / neigh_median_disp
                            
                            if neigh_depth < (my_depth + self.overlap_margin):
                                is_front_fish = False
                                break 

            # --- GUARDAR TODO EN EL OBJETO FISH ---
            fish.aspect_ratio = aspect_ratio
            fish.is_front_fish = is_front_fish 

            # ==========================================
            # 2. APLICAR REGLAS Y TOMAR DECISIÓN
            # ==========================================
            
            cond_aspect_ratio  = (aspect_ratio >= self.aspect_ratio_thr) or self.ignore_aspect_ratio
            cond_completeness  = fish.is_3d_complete or self.ignore_completeness
            cond_borders       = not fish.in_image_borders or self.ignore_borders
            cond_overlap_smart = (not fish.does_overlap) or is_front_fish or self.ignore_overlap
            
            # DECISIÓN DE CÁLCULO 3D (Tiene que cumplir todas las reglas activas)
            fish_3d_ok = cond_aspect_ratio and cond_completeness and cond_borders and cond_overlap_smart
            
            fish.fish_3d_ok = fish_3d_ok
            
            # --- LOGS CLAROS ---
            print(colored(f"\n--- ESTADO FÍSICO FISH {fish.track_id} ---", "cyan"))
            print(f"Aspect Ratio: {aspect_ratio:.2f} | Completo: {fish.is_3d_complete} | Toca Borde: {fish.in_image_borders}")
            print(f"Solapado: {fish.does_overlap} | Está Delante: {fish.is_front_fish}")
            
            if not fish_3d_ok:
                if not cond_aspect_ratio:
                    cprint(f"🚫 Descartado: Está de cara o curvado (Ratio {aspect_ratio:.2f} < {self.aspect_ratio_thr})", "magenta")
                else:
                    cprint(f"🚫 Descartado para medir 3D según tus reglas (ignore_*)", "magenta")
            else:
                cprint(f"✅ Aprobado para cálculo 3D pesado", "green")
                
            return fish_3d_ok