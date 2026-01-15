import os

import numpy as np
import pandas as pd
import open3d as o3d
from typing import List
from pyntcloud import PyntCloud
from sklearn.decomposition import PCA
from scipy import stats
from termcolor import colored

import utils
from fish3D import Fish3D
from fish2D import Fish2D,FrameScene


class Bagfile_fauna():
    def __init__(self,out_path):
        
        self.fish_list = []  
        
        self.out_path = os.path.join(out_path,"results")
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        
        columns_df = ['frame_id','class_name','Object_id','track_id','is_3D_complete','in_image_borders',
                        'fish_direction','elevation_deg','azimuth_deg','raw_length','filtered_length','fish_dist_from_camera']
        
        self.all_fish_df = pd.DataFrame(columns=columns_df)
        self.filtered_result_df = pd.DataFrame()
            
    def add_fish(self, fish: Fish3D):
        """Add a Fish object to the frame"""
        self.fish_list.append(fish)
        fish_data = {
            'frame_id': fish.fish_frame,
            'class_name': fish.class_name,
            'Object_id': fish.color_id,
            'track_id': fish.track_id,
            'is_3D_complete': fish.is_3d_complete,
            'in_image_borders': fish.in_image_borders,
            'fish_direction': fish.fish_direction,
            'elevation_deg': fish.elevation_deg,
            'azimuth_deg': fish.azimuth_deg,
            'raw_length': fish.length,
            'filtered_length': fish.filtered_length,
            'fish_dist_from_camera':  fish.fish_dist_from_camera
            }
            
        self.all_fish_df = pd.concat([self.all_fish_df, pd.DataFrame([fish_data])], ignore_index=True)
    
    def add_fish_2D(self, fish_2d: Fish2D):
        """Add a 2D fish entry when no 3D data (PCD) is available"""
        fish_data = {
            'frame_id': fish_2d.frame_id if hasattr(fish_2d, 'frame_id') else None,
            'class_name': fish_2d.class_name,
            'Object_id': fish_2d.color_id,
            'track_id': fish_2d.track_id,
            'is_3D_complete': fish_2d.is_3d_complete,
            'in_image_borders': fish_2d.in_image_borders,
            'fish_direction': None,
            'elevation_deg': None,
            'azimuth_deg': None,
            'raw_length': None,
            'filtered_length': None,
            'fish_dist_from_camera': None
        }

        self.all_fish_df = pd.concat([self.all_fish_df, pd.DataFrame([fish_data])],ignore_index=True)
    
    def add_single_gt(self,gt):
        self.all_fish_df["gt"] = gt
    
    def process_and_save_df(self):
        
        self.all_fish_df.to_csv(os.path.join(self.out_path,'all_fish_info_raw.csv'))
        
        self.filtered_result_df = self.all_fish_df[
            (self.all_fish_df['is_3D_complete']) &
            (~self.all_fish_df['in_image_borders']) &
            (self.all_fish_df['raw_length'] != -1)
        ]

        self.filtered_df = self.filtered_result_df.copy()
        # self.resume_df = self.filtered_df.groupby('track_id').agg({
        #                 'length': ['mean', 'max'],
        #                 'filtered_length': ['mean', 'max']
        #                 })
        self.resume_df = self.filtered_df.groupby('track_id').agg(
            length_mean=('raw_length', 'mean'),
            length_max=('raw_length', 'max'),
            filtered_length_mean=('filtered_length', 'mean'),
            filtered_length_max=('filtered_length', 'max'),
            entry_count=('track_id', 'count')
        )

    
        self.filtered_df.to_csv(os.path.join(self.out_path, 'all_complete_ok_fish.csv'))
        self.resume_df.to_csv(os.path.join(self.out_path, 'resume.csv'))

        
        
        

            
            
