from cmath import nan
import open3d as o3d
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pyntcloud import PyntCloud
from sklearn.neighbors import KDTree
from scipy import stats
import argparse
import json
import csv
import pandas as pd
from natsort import natsorted
from fish3D import Fish3D
import utils
from bagfile_fauna import Bagfile_fauna
from fish2D import Fish2D,FrameScene
import cloudpickle as pickle
from termcolor import cprint 
# conda activate pointcloud_env
# python /home/uib/DL_utils/ros_dl_ws/src/stereo_plome/scripts/3D_utils/measure_fish.py --base_path /home/uib/DATA/PEIXOS/LANTY/SARMIENTO/Process_and_disparity_x4/out/3D_test/pointclouds/

#### INITIALIZATIONS -----------------------------------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--base_path', default='/home/rosuser/repo/src/out/SARMIENTO/test_2025/07_46_28/pointclouds/')

parsed_args = parser.parse_args()
base_path = parsed_args.base_path

out_path = base_path

use_gt = False
real_measure = 0.315

bagfile_fauna = Bagfile_fauna(out_path)

# fish_dict = utils.load_json_dict(os.path.join(base_path,"model_classes_dict.json"))
# class_colors = utils.load_json_dict(os.path.join(base_path,"class_colours_dict.json"))

# reversed_class_colors = {v: k for k, v in class_colors.items()}
# print("REVERSED CLASS CLOLOURS:")
# print(reversed_class_colors)

frames_dir = base_path.split("pointclouds")[0]  # carpeta donde están los frame*_scene.pkl
pointclouds_dir = base_path

print("hey hey!")

### MAIN LOOP -----------------------------------------------------------------------------------------------------------------------------
# Load a PCD file -> Process each pointcloud
for scene_file in natsorted(os.listdir(frames_dir)):
    if scene_file.endswith("_scene.pkl"):
        frame_id = scene_file.split("_scene.pkl")[0].replace("frame", "")
        frame_scene_path = os.path.join(frames_dir, scene_file)
        pc_file = os.path.join(pointclouds_dir, f"frame{frame_id}_segmented.pcd")
        print("file is ; ",frame_scene_path)
        
        with open(frame_scene_path, "rb") as f:
            scene_object = pickle.load(f)

        print("pc file is; ",pc_file)
        # ¿Existe el pointcloud asociado?
        if os.path.exists(pc_file):
            print(f"✅ Frame {frame_id}: procesando con pointcloud")
            
        # if ".pcd" in pc_file and "color" not in pc_file:
            print(f"✅ Processing frame {frame_id} with pointcloud")

            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            
            pc_id = pc_file.split(".")[0].split("_")[0]
            pc_folder = os.path.join(base_path,pc_id)
            pc_file=os.path.join(base_path,pc_file)
            frame_id = pc_id.split("frame")[-1]

            print("PROCESSING FILE: ",pc_id)
            print("BASE PATH FILE: ",base_path)
            
            with open(frame_scene_path, "rb") as f:
                scene_object = pickle.load(f)
                        
            colors = scene_object.get_fish_colors_from_scene()
            
            print("colors!:",colors)
                
            # Read frame pointcloud info
            pointcloud_info,object_ids_array = utils.read_fish_scene_pc(pc_file)

            objects = list(set(object_ids_array))
            print("objects: ",objects)
            # remove 0 as it is the background object
            if 0 in objects:
                objects.remove(0)

            print("OBJECTS ARE: ",objects)
            for object_id in objects:
                
                current_fish_2d = scene_object.get_fish_from_scene(object_id)
                
                if current_fish_2d is not None:
                    # Retrieve only the current fish points from the scene pointcloud:
                    sub_pointcloud = pointcloud_info[pointcloud_info[:,-1]==object_id][:,:-1]
                    
                    fish_class = current_fish_2d.class_name  
                    track_id = current_fish_2d.track_id  
                    
                    print("object number: ",object_id)
                    print("fish class: ",fish_class)
                    print("Is fish complete? ",current_fish_2d.is_3d_complete)
                    print("IS fish in the borders of the image?",current_fish_2d.in_image_borders)
                
                    
                    current_fish_3d = Fish3D(current_fish_2d, sub_pointcloud,pc_folder)
                    
                    if (current_fish_2d.is_3d_complete==1) and (current_fish_2d.in_image_borders==False):
                        # Filter the fish object outliers
                        # current_fish_3d.filter_outliers()
                        # current_fish_3d.filter_outliers_HDBSCAN()
                        # current_fish_3d.filter_outliers_statistical()
                        # current_fish_3d.filter_outliers_z_diffs()
                        current_fish_3d.filter_outliers_HDBSCAN_adaptive()
                        
                        current_fish_3d.get_distance_camera_fish()

                        raw_ok = current_fish_3d.save_fish_pointcloud(filtered=False)
                        print("SAVED!!!!")
                        filtered_ok = current_fish_3d.save_fish_pointcloud(filtered=True)
                        
                        if raw_ok:
                            current_fish_3d.measure_fish_length_ply_with_angles_and_plot(plot_fish_direction=True,filtered=False)   
                        if filtered_ok:
                            current_fish_3d.measure_fish_length_ply_with_angles_and_plot(plot_fish_direction=True,filtered=True)

                        if current_fish_3d.length!=current_fish_3d.filtered_length:
                            print("A correction has been made!!!")
                    
                    bagfile_fauna.add_fish(current_fish_3d)
        else:
            print(f"⚠️ Frame {frame_id}: sin pointcloud — se añadirá con valores None")
            print(f"⚠️ path to scene {frame_scene_path}: sin pointcloud — se añadirá con valores None")
            with open(frame_scene_path, "rb") as f:
                scene_object = pickle.load(f)
            print("SCENE:",scene_object.fish_list)
            for fish_2d in scene_object.fish_list:
                print("fish yey!! fshhhhhhhhhhhh")
                bagfile_fauna.add_fish_2D(fish_2d)
                    
                
if use_gt:
    bagfile_fauna.add_single_gt(real_measure)

bagfile_fauna.process_and_save_df()
