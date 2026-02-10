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
        """Add a 2D fish entry when no 3D data is available (Corrected)"""
        fish_data = {
            'frame_id': fish_2d.fish_frame if hasattr(fish_2d, 'fish_frame') else None, 
            'class_name': fish_2d.class_name,
            'object_id': fish_2d.color_id,
            'track_id': fish_2d.track_id,
            'is_3D_complete': fish_2d.is_3d_complete,
            'in_image_borders': fish_2d.in_image_borders, 
            'does_overlap': fish_2d.does_overlap,         
            'overlapping_fish_ids': fish_2d.overlapping_ids, 
            
            # Campos 3D vacíos (Correcto)
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
        
        # Reordenar si las columnas existen
        cols_existentes = [c for c in self.columns_order if c in self.all_fish_df.columns]
        self.all_fish_df = self.all_fish_df[cols_existentes]
        
        # Añadir GT si existe
        if hasattr(self, 'gt_value'):
            self.all_fish_df["gt"] = self.gt_value
        
        # Guardar RAW
        self.all_fish_df.to_csv(os.path.join(self.out_path, 'all_fish_info_raw.csv'), index=False)
        
        # 2. Filtrado Básico
        self.filtered_result_df = self.all_fish_df[
            (self.all_fish_df['is_3D_complete'] == True) &
            (self.all_fish_df['does_overlap'] == False) &
            (self.all_fish_df['in_image_borders'] == False) &
            (self.all_fish_df['raw_length'].notna()) &
            (self.all_fish_df['raw_length'] != -1)
        ].copy()
         
        self.filtered_result_df.to_csv(os.path.join(self.out_path, 'all_complete_ok_fish.csv'), index=False)

        # 3. Resumen y Filtrado Avanzado
        if not self.filtered_result_df.empty:
            
            # Resumen estadístico simple
            resume_raw_df = self.filtered_result_df.groupby('track_id').agg(
                length_mean=('raw_length', 'mean'),
                length_max=('raw_length', 'max'),
                entry_count=('track_id', 'count')
            )
            resume_raw_df.to_csv(os.path.join(self.out_path, 'resume_raw.csv'))

            # --- APLICAMOS EL FILTRO INTELIGENTE ---
            self.resume_filtered_df = self._filter_outliers_per_track(
                self.filtered_result_df, 
                min_tracks_abs=3, 
                min_tracks_to_filter=5
            )
            
            self.resume_filtered_df.to_csv(os.path.join(self.out_path, 'resume_filtered_smart.csv'), index=False)
            
            print(f"✅ Resultados guardados en: {self.out_path}")
        else:
            print("⚠️ No quedaron peces válidos tras el filtrado (Complete & Inside Borders).")
    
    def _filter_outliers_per_track(self, df_input, min_tracks_abs=3, min_tracks_to_filter=5):
        """
        Filtra outliers iterativamente comparando el máximo con el siguiente valor.
        """
        resume_list = []
        
        for track_id, track_data in df_input.groupby('track_id'):
            
            # A) Ignorar tracks muy cortos
            if len(track_data) < min_tracks_abs: 
                continue

            # Lista ordenada de mayor a menor
            sorted_lengths = track_data['filtered_length'].sort_values(ascending=False).tolist()
            median_val = np.median(sorted_lengths)
            
            valid_max = sorted_lengths[0] # Por defecto
            
            # --- B) ALGORITMO SMART MAX ---
            # Solo filtramos si hay suficientes datos
            if len(sorted_lengths) >= min_tracks_to_filter:
                
                # Usamos WHILE porque la lista cambia de tamaño dinámicamente
                while len(sorted_lengths) > 2:
                    current_max = sorted_lengths[0]
                    next_max = sorted_lengths[1]
                    current_median = np.median(sorted_lengths) # Recalcular mediana quizás es excesivo, pero seguro
                    
                    # 1. Safety Check (Glitch gigante)
                    if current_max > (current_median * 2.0):
                        sorted_lengths.pop(0)
                        continue

                    # 2. Consistencia (Salto < 5%)
                    diff_percentage = (current_max - next_max) / next_max
                    
                    if diff_percentage < 0.05: 
                        valid_max = current_max # Validado
                        break
                    else:
                        # Salto grande detectado, descartamos este máximo y probamos el siguiente
                        sorted_lengths.pop(0)
                
                # Si nos quedamos sin candidatos en el bucle, cogemos el que quede
                if len(sorted_lengths) <= 2:
                    valid_max = sorted_lengths[0]

            # --- C) Estadísticas ---
            n_top = max(1, int(len(sorted_lengths) * 0.2))
            top_lengths = sorted_lengths[:n_top]
            mean_top_20 = sum(top_lengths) / len(top_lengths)

            stats = {
                'track_id': track_id,
                'class_name': track_data['class_name'].iloc[0],
                'n_frames': len(track_data),
                'max_length_smart': valid_max,
                'mean_top_20_length': mean_top_20,
                'median_length': median_val,
                'dist_camera_mean': track_data['fish_dist_from_camera'].mean()
            }
            resume_list.append(stats)
        
        return pd.DataFrame(resume_list)

        
        

            
            
