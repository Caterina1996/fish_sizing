import os
import numpy as np
import pandas as pd

from typing import List, Dict, Any
from termcolor import colored

from fish_sizing.detection.fish2D import Fish2D, FrameScene 
from fish_sizing.measurement.fish3D import  Fish3D
from fish_sizing.utils import tools


class Bagfile_fauna():
    def __init__(self,out_path,gt=None):
        
        # Buffer para acumular datos
        self.data_buffer: List[Dict[str, Any]] = []
        self.gt = gt
        self.fish_list = []  
        
        self.out_path = os.path.join(out_path,"results")
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        
        # AÑADIDO: Nuevas columnas de error
        self.columns_order = [
        'frame_id', 'class_name', 'object_id', 'track_id', 
        'is_3D_complete', 'in_image_borders','does_overlap',
        'overlapping_fish_ids', 'fish_direction', 'elevation_deg', 'azimuth_deg', 
        'raw_length', 'filtered_length', 'gt', 'abs_error_m', 'rel_error_%', 'fish_dist_from_camera'
        ]
                    
        self.all_fish_df = pd.DataFrame(columns=self.columns_order)
        self.filtered_result_df = pd.DataFrame()
            
    def add_fish(self, fish: Fish3D):
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
            'gt': self.gt,
            'abs_error_m': None, # Se calcula al final
            'rel_error_%': None, # Se calcula al final
            'fish_dist_from_camera':  fish.fish_dist_from_camera
            }
            
        self.data_buffer.append(fish_data)
        
    def add_fish_2D(self, fish_2d: Fish2D):
        fish_data = {
            'frame_id': fish_2d.fish_frame if hasattr(fish_2d, 'fish_frame') else None, 
            'class_name': fish_2d.class_name,
            'object_id': fish_2d.color_id,
            'track_id': fish_2d.track_id,
            'is_3D_complete': fish_2d.is_3d_complete,
            'in_image_borders': fish_2d.in_image_borders, 
            'does_overlap': fish_2d.does_overlap,         
            'overlapping_fish_ids': fish_2d.overlapping_ids, 
            
            'fish_direction': None,
            'elevation_deg': None,
            'azimuth_deg': None,
            'raw_length': None,
            'filtered_length': None,
            'gt': self.gt,
            'abs_error_m': None,
            'rel_error_%': None,
            'fish_dist_from_camera': None
        }
        self.data_buffer.append(fish_data)   
    
    def add_single_gt(self,gt):
        self.gt = gt
        self.all_fish_df["gt"] = gt
    
    def process_and_save_df(self):
        
        if not self.data_buffer:
            print("⚠️ No hay datos de peces para guardar.")
            return

        # 1. Crear DataFrame 
        self.all_fish_df = pd.DataFrame(self.data_buffer)
        
        # --- AÑADIR GT Y CÁLCULO DE ERRORES INDIVIDUALES ---
        if self.gt is not None:
            self.all_fish_df["gt"] = self.gt
            
            # Pasar GT de cm a metros
            gt_m = self.gt / 100.0
            
            # Ignoramos los valores '-1' o nulos para que el error matemático tenga sentido
            valid_lengths = self.all_fish_df['filtered_length'].replace(-1, np.nan).astype(float)
            
            # Cálculo de Errores (Error absoluto y Relativo en %)
            self.all_fish_df["abs_error_m"] = (valid_lengths - gt_m).abs()*100
            self.all_fish_df["rel_error_%"] = (self.all_fish_df["abs_error_m"] / gt_m) * 100

        # Reordenar columnas
        cols_existentes = [c for c in self.columns_order if c in self.all_fish_df.columns]
        self.all_fish_df = self.all_fish_df[cols_existentes]
        
        # Guardar RAW
        self.all_fish_df.to_csv(os.path.join(self.out_path, 'all_fish_info_raw.csv'), index=False)
        
        # 2. Filtrado Básico (Completos, sin solapamiento, dentro de bordes)
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
            
            resume_raw_df = self.filtered_result_df.groupby('track_id').agg(
                length_mean=('raw_length', 'mean'),
                length_max=('raw_length', 'max'),
                entry_count=('track_id', 'count')
            )
            resume_raw_df.to_csv(os.path.join(self.out_path, 'resume_raw.csv'))

            # Filtro Inteligente (Se hereda el cálculo del error sobre el max_length_smart)
            self.resume_filtered_df = self._filter_outliers_per_track(
                self.filtered_result_df, 
                min_tracks_abs=3, 
                min_tracks_to_filter=5
            )
            self.resume_filtered_df.to_csv(os.path.join(self.out_path, 'resume_filtered_smart.csv'), index=False)
            
            # 4. FILTRADO ESTRICTO POR ÁNGULO Z (< 30º)
            angle_threshold = 20.0
            
            df_angle_ok = self.filtered_result_df[
                self.filtered_result_df['elevation_deg'].notna() & 
                (self.filtered_result_df['elevation_deg'].abs() <= angle_threshold)
            ].copy()
            
            df_angle_ok.to_csv(os.path.join(self.out_path, 'all_angle_ok_fish.csv'), index=False)

            if not df_angle_ok.empty:
                self.resume_filtered_angle_df = self._filter_outliers_per_track(
                    df_angle_ok, 
                    min_tracks_abs=3, 
                    min_tracks_to_filter=5
                )
                self.resume_filtered_angle_df.to_csv(os.path.join(self.out_path, 'resume_filtered_smart_angle_ok.csv'), index=False)
                print(f"✅ Resultados extra (Ángulo Z <= {angle_threshold}º) guardados con éxito.")
            else:
                print(f"⚠️ Ningún frame de pez cumplió la condición de ángulo Z <= {angle_threshold}º.")

            print(f"✅ Resultados totales guardados en: {self.out_path}")
        else:
            print("⚠️ No quedaron peces válidos tras el filtrado.")
    
    def _filter_outliers_per_track(self, df_input, min_tracks_abs=3, min_tracks_to_filter=3):
        resume_list = []
        
        for track_id, track_data in df_input.groupby('track_id'):
            
            if len(track_data) < min_tracks_abs: 
                continue

            sorted_lengths = track_data['filtered_length'].sort_values(ascending=False).tolist()
            median_val = np.median(sorted_lengths)
            valid_max = sorted_lengths[0] 
            
            if len(sorted_lengths) >= min_tracks_to_filter:
                while len(sorted_lengths) > 2:
                    current_max = sorted_lengths[0]
                    next_max = sorted_lengths[1]
                    current_median = np.median(sorted_lengths) 
                    
                    if current_max > (current_median * 2.0):
                        sorted_lengths.pop(0)
                        continue

                    diff_percentage = (current_max - next_max) / next_max
                    
                    if diff_percentage < 0.05: 
                        valid_max = current_max 
                        break
                    else:
                        sorted_lengths.pop(0)
                
                if len(sorted_lengths) <= 2:
                    valid_max = sorted_lengths[0]

            n_top = max(1, int(len(sorted_lengths) * 0.2))
            top_lengths = sorted_lengths[:n_top]
            mean_top_20 = sum(top_lengths) / len(top_lengths)

            # --- CÁLCULO DE ERRORES DEL RESUMEN AGREGADO ---
            abs_err = None
            rel_err = None
            
            if self.gt is not None and self.gt > 0:
                gt_m = self.gt / 100.0
                abs_err = abs(valid_max - gt_m) * 100 # el vull en cm
                rel_err = (abs_err / gt_m) * 100

            stats = {
                'track_id': track_id,
                'class_name': track_data['class_name'].iloc[0],
                'n_frames': len(track_data),
                'max_length_smart': valid_max,
                'mean_top_20_length': mean_top_20,
                'median_length': median_val,
                'gt': self.gt, 
                'abs_error_m': abs_err,       # Nuevo
                'rel_error_%': rel_err,       # Nuevo
                'dist_camera_mean': track_data['fish_dist_from_camera'].mean()
            }
            resume_list.append(stats)
        
        return pd.DataFrame(resume_list)