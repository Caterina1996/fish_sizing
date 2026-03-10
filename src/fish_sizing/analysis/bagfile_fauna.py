import os
import numpy as np
import pandas as pd

from typing import List, Dict, Any
from termcolor import colored

from fish_sizing.detection.fish2D import Fish2D, FrameScene 
from fish_sizing.measurement.fish3D import  Fish3D
from fish_sizing.utils import tools


class Bagfile_fauna():
    def __init__(self, out_path, gt=None):
        
        self.data_buffer: List[Dict[str, Any]] = []
        self.gt = gt
        self.fish_list = []  
        
        self.out_path = os.path.join(out_path,"results")
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        
        self.columns_order = [
            'frame_id', 'class_name', 'object_id', 'track_id', 
            'is_3D_complete', 'in_image_borders', 'does_overlap', 'is_front_fish', 
            'aspect_ratio', 'pointcloud_size_ok','fish_3d_ok', 'overlapping_fish_ids', 
            'fish_direction', 'elevation_deg', 'azimuth_deg', 
            'raw_length', 'filtered_length', 'spine_length', 
            'gt', 'abs_error_cm', 'rel_error_%', 'fish_dist_from_camera'
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
            'is_3D_complete': getattr(fish, 'is_3d_complete', False),
            'in_image_borders': getattr(fish, 'in_image_borders', False),
            'does_overlap': getattr(fish, 'does_overlap', False),
            'overlapping_fish_ids': getattr(fish, 'overlapping_ids', []),
            "aspect_ratio": getattr(fish, 'aspect_ratio', 0.0),
            "is_front_fish": getattr(fish, 'is_front_fish', True),
            "fish_3d_ok": getattr(fish, 'fish_3d_ok', False),
            'fish_direction': getattr(fish, 'fish_direction', None),
            'elevation_deg': getattr(fish, 'elevation_deg', None),
            'azimuth_deg': getattr(fish, 'azimuth_deg', None),
            'raw_length': getattr(fish, 'length', -1),
            'filtered_length': getattr(fish, 'filtered_length', -1),
            'spine_length': getattr(fish, 'spine_length', -1),
            'gt': self.gt,
            'pointcloud_size_ok': getattr(fish, 'pointcloud_size_ok', False),
            'abs_error_cm': None, 
            'rel_error_%': None,  
            'fish_dist_from_camera': getattr(fish, 'fish_dist_from_camera', -1)
        }
        self.data_buffer.append(fish_data)
        
    def add_fish_2D(self, fish_2d: Fish2D):
        fish_data = {
            'frame_id': fish_2d.fish_frame if hasattr(fish_2d, 'fish_frame') else None, 
            'class_name': fish_2d.class_name,
            'object_id': fish_2d.color_id,
            'track_id': fish_2d.track_id,
            'is_3D_complete': getattr(fish_2d, 'is_3d_complete', False),
            'in_image_borders': getattr(fish_2d, 'in_image_borders', False), 
            'does_overlap': getattr(fish_2d, 'does_overlap', False),         
            'overlapping_fish_ids': getattr(fish_2d, 'overlapping_ids', []), 
            "aspect_ratio": getattr(fish_2d, 'aspect_ratio', 0.0),
            "is_front_fish": getattr(fish_2d, 'is_front_fish', True),
            "fish_3d_ok": getattr(fish_2d, 'fish_3d_ok', False),
            'fish_direction': None,
            'elevation_deg': None,
            'azimuth_deg': None,
            'raw_length': None,
            'filtered_length': None,
            'spine_length': None,
            'gt': self.gt,
            'pointcloud_size_ok': None,
            'abs_error_cm': None,
            'rel_error_%': None,
            'fish_dist_from_camera': None
        }
        self.data_buffer.append(fish_data)   
    
    def add_single_gt(self, gt):
        self.gt = gt
        self.all_fish_df["gt"] = gt
    
    def process_and_save_df(self):
        if not self.data_buffer:
            print("⚠️ No hay datos de peces para guardar.")
            return

        self.all_fish_df = pd.DataFrame(self.data_buffer)
        
        if self.gt is not None:
            self.all_fish_df["gt"] = self.gt
            gt_m = self.gt / 100.0 
            valid_lengths = self.all_fish_df['filtered_length'].replace(-1, np.nan).astype(float)
            self.all_fish_df["abs_error_cm"] = (valid_lengths - gt_m).abs() * 100
            self.all_fish_df["rel_error_%"] = (self.all_fish_df["abs_error_cm"] / self.gt) * 100

        cols_existentes = [c for c in self.columns_order if c in self.all_fish_df.columns]
        self.all_fish_df = self.all_fish_df[cols_existentes]
        self.all_fish_df.to_csv(os.path.join(self.out_path, 'all_fish_info_raw.csv'), index=False)
        
        # FILTRADO ESTRICTO
        self.filtered_result_df = self.all_fish_df[
            (self.all_fish_df['is_3D_complete'] == True) &
            (self.all_fish_df['in_image_borders'] == False) &
            (self.all_fish_df['does_overlap'] == False) &
            (self.all_fish_df['is_front_fish'] == True) &
            (self.all_fish_df['aspect_ratio'] >= 1.8) & 
            (self.all_fish_df['filtered_length'].notna()) &
            (self.all_fish_df['filtered_length'] > 0)
        ].copy()
         
        self.filtered_result_df.to_csv(os.path.join(self.out_path, 'all_complete_ok_fish.csv'), index=False)

        if not self.filtered_result_df.empty:
            resume_raw_df = self.filtered_result_df.groupby('track_id').agg(
                length_mean=('raw_length', 'mean'),
                length_max=('raw_length', 'max'),
                entry_count=('track_id', 'count')
            )
            resume_raw_df.to_csv(os.path.join(self.out_path, 'resume_raw.csv'))

            self.resume_filtered_df = self._filter_outliers_per_track(
                self.filtered_result_df, 
                min_tracks_abs=3, 
                min_tracks_to_filter=5
            )
            self.resume_filtered_df.to_csv(os.path.join(self.out_path, 'resume_filtered_smart.csv'), index=False)
            
            # FILTRADO POR ÁNGULO (Parte correctamente de filtered_result_df)
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
                print(f"✅ Resultados extra (Ángulo Z <= {angle_threshold}º) guardados.")
            else:
                print(f"⚠️ Ningún pez cumplió la condición de ángulo Z <= {angle_threshold}º.")

            print(f"✅ Resultados totales guardados en: {self.out_path}")
        else:
            print("⚠️ No quedaron peces válidos tras el filtrado estricto.")
    
    def _filter_outliers_per_track(self, df_input, min_tracks_abs=3, min_tracks_to_filter=3):
        resume_list = []
        for track_id, track_data in df_input.groupby('track_id'):
            num_frames = len(track_data)
            if num_frames < min_tracks_abs: 
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

            if num_frames < 20:
                mean_top_20 = valid_max  
                representative_length = valid_max  
            else:
                n_top = max(1, int(len(sorted_lengths) * 0.2))
                top_lengths = sorted_lengths[:n_top]
                mean_top_20 = sum(top_lengths) / len(top_lengths)
                representative_length = mean_top_20

            abs_err = None
            rel_err = None
            if self.gt is not None and self.gt > 0:
                gt_m = self.gt / 100.0
                abs_err = abs(representative_length - gt_m) * 100 
                rel_err = (abs_err / self.gt) * 100 
                
            sorted_spine = track_data['spine_length'].dropna().sort_values(ascending=False).tolist()
            median_spine = np.median(sorted_spine) if sorted_spine else None
            max_spine = sorted_spine[0] if sorted_spine else None
            mean_elevation = track_data['elevation_deg'].mean()

            # Recuperadas las columnas booleanas de resumen
            track_had_overlap = track_data['does_overlap'].any()
            track_touched_borders = track_data['in_image_borders'].any()

            stats = {
                'track_id': track_id,
                'class_name': track_data['class_name'].iloc[0],
                'n_frames_validos': num_frames,
                'does_overlap': track_had_overlap,       
                'in_image_borders': track_touched_borders, 
                'max_length_smart': valid_max,
                'mean_top_20_length': mean_top_20,
                'length_used_for_error': representative_length, 
                'median_length': median_val,
                'max_spine_length': max_spine,       
                'median_spine_length': median_spine, 
                'mean_elevation_deg': mean_elevation, 
                'gt': self.gt, 
                'abs_error_cm': abs_err,       
                'rel_error_%': rel_err,       
                'dist_camera_mean': track_data['fish_dist_from_camera'].mean()
            }
            resume_list.append(stats)
        
        return pd.DataFrame(resume_list)