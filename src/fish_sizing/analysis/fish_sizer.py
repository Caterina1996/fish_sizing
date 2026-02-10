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
            
        self.scene_ply_name = os.path.join(self.out_path, f"{self.frame_scene.frame_name}_scene.ply")
        self.all_fish_ply_name = os.path.join(self.out_path, f"{self.frame_scene.frame_name}_all_fish.ply")
    
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
                
                # Save fish pc
                fish_ply_path = os.path.join(self.out_path, f"{self.frame_scene.frame_name}_{fish.color_id}.ply")
                fish_pcd = self.stereo.extract_point_cloud(self.frame_scene.scene_points_3d, self.img_l, mask=fish.mask)
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
                    
                    # A) Filtrado y Medición 3D
                    current_fish_3d.filter_outliers_HDBSCAN_adaptive()
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
            
    def check_3d(self,fish):            
                
        is_front_fish = True # Por defecto asumimos que sí
            
        if fish.does_overlap and not self.ignore_overlap:
            # Vamos a comprobar si somos el pez de delante
            
            # Usamos la máscara para sacar solo la disparidad del pez
            my_disp_values = self.frame_scene.disparity_map[fish.mask > 0]
            
            if len(my_disp_values) > 0:
                my_median_disp = np.median(my_disp_values)
            else:
                my_median_disp = 0

            # 2. Comprobar solapamiento con otros peces
            for neighbour_id in fish.overlapping_ids:
                # Buscamos al vecino en la lista de la escena
                neighbour = next((f for f in self.frame_scene.fish_list if f.track_id == neighbour_id), None)
                
                if neighbour and neighbour.mask is not None:
                    neighbour_disp_values = self.frame_scene.disparity_map[neighbour.mask > 0]
                    if len(neighbour_disp_values) > 0:
                        neigh_median_disp = np.median(neighbour_disp_values)

                        # Evitamos división por cero
                        if my_median_disp > 0.1 and neigh_median_disp > 0.1:
                            my_depth = (self.stereo.FOCAL * self.stereo.BASELINE) / my_median_disp
                            neigh_depth = (self.stereo.FOCAL * self.stereo.BASELINE) / neigh_median_disp
                            
                        # 3.CHECK DIFF IN Zs: Para ser el "Front Fish", mi profundidad debe ser menor (más cerca)
                        # que la del vecino MENOS el margen.
                        # Es decir: quiero que neighbour_Z > My_Z + Margen
                        if neigh_depth < (my_depth + self.overlap_margin):
                            is_front_fish = False
                            cprint(f"   🚫 Fish {self.fish.track_id} descartado: Está detrás o pegado al Fish {neighbour_id}", "magenta")
                            break # Ya no hace falta mirar más, estoy ocluido.  

            # Ahora actualizamos la condición final
            cond_completeness = fish.is_3d_complete or self.ignore_completeness
            cond_borders = not fish.in_image_borders or self.ignore_borders
            
            # Aceptamos si NO hay solapamiento O SI hay solapamiento pero somos el de delante
            cond_overlap_smart = (not fish.does_overlap) or (is_front_fish) or self.ignore_overlap
            
            fish_3d_ok = cond_completeness and cond_borders and cond_overlap_smart
            
            return fish_3d_ok
