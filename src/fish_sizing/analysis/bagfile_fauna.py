import os
import numpy as np
import pandas as pd

from typing import List
from termcolor import colored

from fish_sizing.detection.fish2D import Fish2D, FrameScene 
from fish_sizing.measurement.fish3D import  Fish3D
from fish_sizing.utils import tools


class Bagfile_fauna():
    def __init__(self,out_path):
        
        # Buffer para acumular datos (Lista de diccionarios)
        # + rápido que concatenar DataFrames en cada iteración
        self.data_buffer: List[Dict[str, Any]] = []
        
        self.fish_list = []  
        
        self.out_path = os.path.join(out_path,"results")
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        
        self.columns_order = [
        'frame_id', 'class_name', 'object_id', 'track_id', 
        'is_3D_complete', 'in_image_borders','does_overlap',
        'overlapping_fish_ids', 'fish_direction', 'elevation_deg', 'azimuth_deg', 
        'raw_length', 'filtered_length', 'fish_dist_from_camera'
        ]
                    
        self.all_fish_df = pd.DataFrame(columns=self.columns_order)
        self.filtered_result_df = pd.DataFrame()
            
    def add_fish(self, fish: Fish3D):
        """Add a Fish object to the frame"""
        self.fish_list.append(fish)
        fish_data = {
            'frame_id': fish.fish_frame,
            'class_name': fish.class_name,
            'object_id': fish.color_id,
            'track_id': fish.track_id,
            'is_3D_complete': fish.is_3d_complete,
            'in_image_borders': fish.in_image_borders,
            'does_overlap': fish.does_overlap,
            'overlapping_fish_ids': fish.overlapping_ids,
            'fish_direction': fish.fish_direction,
            'elevation_deg': fish.elevation_deg,
            'azimuth_deg': fish.azimuth_deg,
            'raw_length': fish.length,
            'filtered_length': fish.filtered_length,
            'fish_dist_from_camera':  fish.fish_dist_from_camera
            }
            
        self.data_buffer.append(fish_data)
        
    def add_fish_2D(self, fish_2d: Fish2D):
        """Add a 2D fish entry when no 3D data (PCD) is available"""
        fish_data = {
            'frame_id': fish_2d.frame_id if hasattr(fish_2d, 'frame_id') else None,
            'class_name': fish_2d.class_name,
            'object_id': fish_2d.color_id,
            'track_id': fish_2d.track_id,
            'is_3D_complete': fish_2d.is_3d_complete,
            'in_image_borders': fish_2d.in_image_borders,
            'in_image_borders': None,
            'does_overlap': None,
            'fish_direction': None,
            'elevation_deg': None,
            'azimuth_deg': None,
            'raw_length': None,
            'filtered_length': None,
            'fish_dist_from_camera': None
        }

        self.data_buffer.append(fish_data)    
    
    def add_single_gt(self,gt):
        self.all_fish_df["gt"] = gt
    
    def process_and_save_df(self):
        
        if not self.data_buffer:
            print("⚠️ No hay datos de peces para guardar.")
            return

        # 1. Crear DataFrame 
        self.all_fish_df = pd.DataFrame(self.data_buffer)
        self.all_fish_df = self.all_fish_df[self.columns_order]
        
        # Añadir GT si existe
        if hasattr(self, 'gt_value'):
            self.all_fish_df["gt"] = self.gt_value
        
        self.all_fish_df.to_csv(os.path.join(self.out_path,'all_fish_info_raw.csv'))
        
        self.filtered_result_df = self.all_fish_df[
            (self.all_fish_df['is_3D_complete'] == True) &
            (self.all_fish_df['does_overlap'] == False) &
            (self.all_fish_df['in_image_borders'] == False) &
            (self.all_fish_df['raw_length'].notna()) &  # Mejor que != -1 para manejar Nones de 2D
            (self.all_fish_df['raw_length'] != -1)]
         
        self.filtered_df = self.filtered_result_df.copy()
        self.filtered_result_df.to_csv(os.path.join(self.out_path, 'all_complete_ok_fish.csv'), index=False)

             # 3. Resumen (Group By Track ID)
        if not self.filtered_result_df.empty:
            self.resume_df = self.filtered_result_df.groupby('track_id').agg(
                length_mean=('raw_length', 'mean'),
                length_max=('raw_length', 'max'),
                length_std=('raw_length', 'std'), # Útil para ver estabilidad
                filtered_length_mean=('filtered_length', 'mean'),
                filtered_length_max=('filtered_length', 'max'),
                entry_count=('track_id', 'count')
            )
            
            # Guardar Filtrados y Resumen
            self.filtered_result_df.to_csv(os.path.join(self.out_path, 'all_complete_ok_fish.csv'), index=False)
            self.resume_df.to_csv(os.path.join(self.out_path, 'resume.csv'))
            
            print(f"✅ Resultados guardados en: {self.out_path}")
        else:
            print("⚠️ No quedaron peces válidos tras el filtrado (Complete & Inside Borders).")


        
        

            
            
