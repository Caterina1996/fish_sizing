
import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
from termcolor import cprint
from fish_sizing.utils.config import PROCESSING_PIPELINES
import yaml
from pathlib import Path
import logging
from termcolor import colored
import shutil
from natsort import natsorted

#  Default tereo camera config
# baseline_m = abs(-179.77544 / 1469.28052)
# left_cam_pos = np.array([0, 0, 0])
# right_cam_pos = np.array([baseline_m, 0, 0])
# cam_dir = np.array([0, 0, 1])

# Global variable inside the module to keep the logger alive
run_logger = None

######################################################
#               I/O TOOLS 
#######################################################

def load_json_dict(dict_path):
    try:
        # Opening JSON file
        with open(dict_path) as json_file:
            json_dict = json.load(json_file)
        print("Dict ", dict_path,"is: ",json_dict)
    except Exception as e:
        print("WARNING!! : Could not read json Dict. Check that the file exists:",dict_path)
        return {}

    return json_dict

def setup_logger(out_dir):
    """Configures the logger to write to out_dir/execution_log.txt"""
    global run_logger
    run_logger = logging.getLogger("FishSizingLogger")
    run_logger.setLevel(logging.DEBUG)
    
    # Clear previous handlers in case we run in a loop
    if run_logger.hasHandlers():
        run_logger.handlers.clear()
        
    # Create the file handler (Clean text)
    log_path = os.path.join(out_dir, "execution_log.txt")
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    
    # Format: [2025-08-21 13:45:00] [ERROR] 💥 Message...
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    run_logger.addHandler(file_handler)

def cprint_and_log(msg, color=None, attrs=None, level=logging.INFO):
    """Prints in color to the terminal AND saves the clean text to log.txt"""
    # 1. Terminal (With colors)
    print(colored(msg, color, attrs=attrs))
    
    # 2. File (Clean text)
    if run_logger:
        run_logger.log(level, msg)

def save_run_config(out_dir, args, conf_thr, gt, visualize_online, use_wls, image_channels):
    """
    Saves the configuration in YAML. 
    STRICT: Intentionally crashes if vital arguments are missing, 
    leaving a trace in the log before dying.
    """
    # 0. Initialize the logger for this folder
    setup_logger(out_dir)
    
    cprint_and_log("Starting YAML configuration dump...", "cyan", level=logging.INFO)

    # ==========================================
    # GUARDS (FAIL-FAST) WITH LOGGING
    # ==========================================
    if not hasattr(args, 'stereo_config'):
        msg = "💥 FATAL ERROR: The script did not receive 'stereo_config' in args. Cannot save the log blindly!"
        cprint_and_log(msg, "red", ["bold"], level=logging.CRITICAL)
        raise ValueError(msg)
        
    if not hasattr(args, 'model_path'):
        msg = "💥 FATAL ERROR: Missing 'model_path' in args. Which YOLO model are you using?"
        cprint_and_log(msg, "red", ["bold"], level=logging.CRITICAL)
        raise ValueError(msg)
        
    if not hasattr(args, 'selected_pipeline'):
        msg = "💥 FATAL ERROR: Missing 'selected_pipeline' in args. I won't know which image processing you saved."
        cprint_and_log(msg, "red", ["bold"], level=logging.CRITICAL)
        raise ValueError(msg)

    # 1. Read the original stereo configuration file
    stereo_cfg = {}
    if args.stereo_config:
        if not os.path.exists(args.stereo_config):
            msg = f"💥 FATAL ERROR: The stereo_config file does not exist at {args.stereo_config}"
            cprint_and_log(msg, "red", ["bold"], level=logging.CRITICAL)
            raise FileNotFoundError(msg)
            
        with open(args.stereo_config, 'r') as f:
            stereo_cfg = yaml.safe_load(f)
            
    # 2. Gather global configuration
    globals_cfg = {
        "MODEL_PATH": args.model_path,
        "CONF_THR": conf_thr,
        "gt_ground_truth": gt,
        "Visualize_online": visualize_online,
        "use_wls": use_wls,
        "image_channels": image_channels,
        "selected_pipeline_name": args.selected_pipeline
    }
    
    # 3. Get the image pipeline
    img_pipeline_steps = PROCESSING_PIPELINES.get(args.selected_pipeline, [])
    if not img_pipeline_steps and args.selected_pipeline != "raw":
        cprint_and_log(f"⚠️ WARNING: The pipeline '{args.selected_pipeline}' does not exist in config.py", "yellow", level=logging.WARNING)
        
    pipeline_readable = [{"step": step[0], "params": step[1], "enabled_or_debug": step[2]} for step in img_pipeline_steps]

    # 4. Group everything
    full_config = {
        "execution_args": vars(args),
        "global_variables": globals_cfg,
        "image_processing_pipeline": pipeline_readable,
        "stereo_configuration": stereo_cfg
    }
    
    # 5. Save to disk
    config_path = os.path.join(out_dir, "run_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)
        
    cprint_and_log(f"📄 Configuration file saved at: {config_path}", "green", level=logging.INFO)
    
def move_inferred_images(out_path):
    """Moves all *_inferred.* images to an _inferred/ subfolder"""
    inferred_dir = os.path.join(out_path, "_inferred")
    os.makedirs(inferred_dir, exist_ok=True)
    
    out_p = Path(out_path)
    moved_count = 0
    
    # Buscar en toda la carpeta de salida
    for file_path in out_p.rglob("*_inferred*.*"):
        # Ignorar si ya está dentro de la carpeta _inferred
        if "_inferred" in file_path.parent.parts:
            continue
        
        if file_path.is_file():
            dest_path = os.path.join(inferred_dir, file_path.name)
            shutil.move(str(file_path), dest_path)
            moved_count += 1
            
    if moved_count > 0:
        cprint(f"✅ Moved {moved_count} images to folder _inferred/", "green")
        
def stream_stereo_from_folder(folder_path):
    """
    Generator that yields stereo pairs from a folder.
    Matches files containing 'left' with their 'right' counterparts.
    """
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    
    try:
        all_files = os.listdir(folder_path)
    except FileNotFoundError:
        cprint_and_log(f"❌ Error: Folder not found: {folder_path}", "red", level=logging.ERROR)
        return

    # Filter only left images (case insensitive)
    left_files = [f for f in all_files if "left" in f.lower() and f.lower().endswith(valid_exts)]
    left_files = natsorted(left_files)
    
    cprint_and_log(f"📂 Found {len(left_files)} image pairs in {folder_path}", "cyan")

    for f_left in left_files:
        # Safer replacement: only replace the filename part, not the whole path
        f_right = f_left.lower().replace("left", "right")
                
        path_l = os.path.join(folder_path, f_left)
        path_r = os.path.join(folder_path, f_right)
        
        if not os.path.exists(path_r):
            cprint_and_log(f"⚠️ Warning: Right pair not found for {f_left}. Skipping.", "yellow", level=logging.WARNING)
            continue
            
        img_l = cv2.imread(path_l)
        img_r = cv2.imread(path_r)
        
        if img_l is None or img_r is None:
            cprint_and_log(f"❌ Error reading images: {f_left}", "red", level=logging.ERROR)
            continue
            
        # Extract ID without extension
        frame_id = os.path.splitext(f_left)[0]
        yield frame_id, img_l, img_r

######################################################
#               2D MASK UTILS
#######################################################

def find_mask_length(self, image, object_id,disp_or_mask="disp"):

    length=-1
    ellipse=None

    # Find contours of the mask (BE CAREFUL! In previous versions of opencv it returns just two values!!)
    contours,_ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    if contours not in [(), []]: # check it's not empty

        #  Merge the contours into a single contour (disp can be divided)
        contour = np.concatenate(contours)
        
        if self.debug_mode:
            cv2.drawContours(image_rgb, contours, -1, (0,255,0), thickness=1)
            cv2.imwrite(os.path.join(self.debug_path,str(self.image_id)+"_object"+str(object_id)+"_contourns_"+disp_or_mask+".png"), image_rgb)
    

        if contour.shape[0] > 5: # si no hi ha 5 punts no pot fitar-hi una elipse
            # Fit an ellipse to the contour
            try:
                hull = cv2.convexHull(contour).reshape(-1, 1, 2) # use hull to have a better ellipse (when mask is disconnected for example)
                ellipse = cv2.fitEllipse(hull)
    
                if self.debug_mode:
                    cv2.polylines(image_rgb, [hull], isClosed=True, color=(255, 0, 0), thickness=2)
                    cv2.imwrite(os.path.join(self.debug_path,self.image_id+"_object"+str(object_id)+ "_HULL of contour "+disp_or_mask +".png"), image_rgb)
                    cv2.ellipse(image_rgb, ellipse, (0, 0, 255), 2)
                    cv2.imwrite(os.path.join(self.debug_path,self.image_id +"_object"+str(object_id)+ "_Ellipse of contour "+disp_or_mask +".png"), image_rgb)
                    
                # Calculate the length of the fish as the major axis length
                #ellipse returns (center,axis,rotation): center->[x,y], axis->[length major axis, length minor axis], rotation angle relative to the hzt counterclkws
                length = max(ellipse[1])
            except Exception as e:
                rospy.logwarn("SOMETHING WRONG WITH THE ELLIPSE: %s",e)

        else:
            print("OBJECT ID: ",object_id)
            print("INCOMPLETE CONTOUR: ",contour.shape)
            print("Incomplete contour of size:",len(contour))
            cprint(f"Incomplete contour of size {len(contour[0])}", "red")

    return length, ellipse

################################################################
#               POINTCLOUD UTILS 
# #############################################################3333333

def create_pointcloud(sub_pointcloud):
    """Create an Open3D PointCloud from a N x 4 array (x, y, z, intensity)."""
    x_array, y_array, z_array,intensity_array = sub_pointcloud[:,0],sub_pointcloud[:,1],sub_pointcloud[:,2],sub_pointcloud[:,3]
    # Create an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.column_stack((x_array, y_array, z_array)))
    pcd.colors = o3d.utility.Vector3dVector(np.column_stack((intensity_array / 255.0, intensity_array / 255.0, intensity_array / 255.0)))
    # Set the colors based on intensity, using red and blue channels
    pcd.colors = o3d.utility.Vector3dVector(np.column_stack((intensity_array / 255.0, np.zeros_like(intensity_array), 1.0 - (intensity_array / 255.0))))
    return pcd

def read_fish_scene_pc(pc_file):
    x_values,y_values,z_values,intensity_values,object_ids = [],[],[],[],[]

    # Read the PCD file and parse its content starting from line 12
    with open(pc_file, 'r') as file:
        lines = file.readlines()[11:]  # Start reading from line 12
        for line in lines:
            values = line.split()
            if len(values) == 5:
                if int(values[3]) > 0:
                    x_values.append(float(values[0]))
                    y_values.append(float(values[1]))
                    z_values.append(float(values[2]))
                    intensity_values.append(int(values[3]))
                    object_ids.append(int(values[4]))

    # Convert lists to NumPy arrays
    x_array = np.array(x_values)
    y_array = np.array(y_values)
    z_array = np.array(z_values)
    intensity_array = np.array(intensity_values)
    object_ids_array = np.array(object_ids)
    print("Intensity set is: ",set(intensity_array))
    print("number of interesting fish: ",set(object_ids_array))

    # Cada camp com una columna
    pointcloud_info=np.stack((x_array,y_array,z_array,intensity_array,object_ids_array),axis=1)
    
    return pointcloud_info,object_ids_array

###############################################################
# VISUALIZATION TOOLS 
# ###############################################################

def draw_camera(ax, origin, direction, cone_height=0.05, cone_radius=0.02, color='black'):
    n = 20
    theta = np.linspace(0, 2 * np.pi, n)
    circle_x = cone_radius * np.cos(theta)
    circle_y = cone_radius * np.sin(theta)
    circle_z = np.zeros_like(circle_x)
    tip = origin + direction * cone_height
    base = np.vstack((circle_x, circle_y, circle_z)).T
    faces = [[tip, origin + base[i], origin + base[(i + 1) % n]] for i in range(n)]
    cone = Poly3DCollection(faces, color=color, alpha=0.5)
    ax.add_collection3d(cone)
    ax.scatter(*origin, color=color, s=30)


def add_arrow(fig, start, direction, color, name, scale=1.0):
    end = start + direction * scale
    fig.add_trace(go.Scatter3d(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        z=[start[2], end[2]],
        mode='lines',
        line=dict(color=color, width=5),
        name=name
    ))

def plot_fish_with_dual_cameras_plotly(points, direction, length, object_id, out_path, filtered=True,
                                        left_cam_pos=None, right_cam_pos=None, cam_dir=None):
    centroid = np.mean(points, axis=0)
    out_dir = os.path.join(out_path, "results")
    os.makedirs(out_dir, exist_ok=True)

    fig = go.Figure()

    # 1. Dibujar puntos del pez
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=5, color='rgba(30, 144, 255, 0.7)'), # color blau
        name=f"Fish {object_id}"
    ))

    # 2. Flecha de la direccion del pez
    direction_vector = direction / (np.linalg.norm(direction) + 1e-12)
    half_length = length / 2.0
    start_position = centroid - direction_vector * half_length
    add_arrow(fig, start_position, direction_vector, color='red', name='Fish Direction', scale=length)

    # 3. Dibujar cámaras si están disponibles
    if left_cam_pos is not None and right_cam_pos is not None and cam_dir is not None:
        for cam_pos, cam_color, name in zip(
            [left_cam_pos, right_cam_pos],
            ['blue', 'green'],
            ['Left Camera', 'Right Camera']
        ):
            fig.add_trace(go.Scatter3d(
                x=[cam_pos[0]], y=[cam_pos[1]], z=[cam_pos[2]],
                mode='markers',
                marker=dict(size=5, color=cam_color),
                name=name
            ))
            add_arrow(fig, cam_pos, cam_dir, color=cam_color, name=f"{name} View", scale=0.05)

    # 4. Dibujar ejes XYZ
    axis_len = 0.1
    add_arrow(fig, np.array([0, 0, 0]), np.array([1, 0, 0]), 'cyan', 'X Axis', scale=axis_len)
    add_arrow(fig, np.array([0, 0, 0]), np.array([0, 1, 0]), 'magenta', 'Y Axis', scale=axis_len)
    add_arrow(fig, np.array([0, 0, 0]), np.array([0, 0, 1]), 'yellow', 'Z Axis', scale=axis_len)

    # 5. Diagonal de la bb que contiene el pez
    # fish_length_measured = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    # print(f"Fish measured length: {fish_length_measured:.3f} m")

    # 6. Añadir texto en 3D con la medida
    fig.add_trace(go.Scatter3d(
        x=[centroid[0]],
        y=[centroid[1]],
        z=[centroid[2]],
        mode='text',
        text=[f"Length: {length:.3f} m"],
        textposition='top center',
        textfont=dict(size=16, color='black'),
        showlegend=True
    ))

    # 7. Ajustar zoom para incluir pez + cámaras + origen
    all_positions = [points, np.array([[0, 0, 0]])]
    if left_cam_pos is not None:
        all_positions.append(left_cam_pos.reshape(1, 3))
    if right_cam_pos is not None:
        all_positions.append(right_cam_pos.reshape(1, 3))

    all_positions = np.vstack(all_positions)
    padding = 0.05
    x_range = [all_positions[:, 0].min() - padding, all_positions[:, 0].max() + padding]
    y_range = [all_positions[:, 1].min() - padding, all_positions[:, 1].max() + padding]
    z_range = [all_positions[:, 2].min() - padding, all_positions[:, 2].max() + padding]

    fig.update_layout(
        title=f"Fish {object_id} with Stereo Camera Setup",
        scene=dict(
            xaxis=dict(title='X', range=x_range),
            yaxis=dict(title='Y', range=y_range),
            zaxis=dict(title='Z', range=z_range),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    html_path = os.path.join(out_dir, f"fish_{object_id}.html")
    fig.write_html(html_path)
    cprint(f"✅ Saved interactive 3D plot to: {html_path}","cyan")

    return html_path