import os
from ultralytics import YOLO
from termcolor import cprint

# --- CONFIGURACIÓN ---
PATH_MAPPINGS = {
    "/home/slimbook/bagfiles": "/home/rosuser/dataset/bagfiles",
    "/home/slimbook/fish_sizing/out": "/home/rosuser/repo/out",
    "/home/slimbook/models": "/home/rosuser/dataset/models/",
    "home/slimbook/fish_sizing/config" :"/home/rosuser/repo/config/"
}

USE_DOCKER = True

conf_thr = 0.5

IN_PATH="/home/slimbook/fish_sizing/out/overfitting_dataset/images/"
# BAGFILE_PATH="//home/slimbook/bagfiles/LIMA/2025/2025_08_21/test_comprsesion/13_34_24/stereo_camera_images_2025-08-21-13-34-25_0.bag"


# MODEL_PATH = "/home/slimbook/models/yv11l/ylarge_d18_poolv2r_lantytr_nocturnes/weights/best.pt"
MODEL_PATH = "/home/slimbook/models/binary/yv11m/Pool_new_2025_from_ckpt/50eps_from_ckpt_of_dbinary_poolv2r_d2_lantytr_nocturnes/weights/best.pt"

# OUT_PATH = "/home/slimbook/fish_sizing/out/test_export/2025-05-08-11-18-25_1/"
OUT_PATH = "/home/slimbook/fish_sizing/out/overfitting_dataset/images/"

# --- FUNCIONES AUXILIARES ---

def transform_path2docker(path: str) -> str:
    """Transform a path from local computer to docker structure."""
    if not USE_DOCKER or path is None:
        return path

    for host_path, docker_path in PATH_MAPPINGS.items():
        if host_path in path:
            new_path = path.replace(host_path, docker_path)
            cprint(f"🔄 Path mapped: {path} \n   -> {new_path}", "yellow")
            return new_path 
            
    return path

if USE_DOCKER:
    MODEL_PATH = transform_path2docker(MODEL_PATH)
    IN_PATH = transform_path2docker(IN_PATH)
    OUT_PATH = transform_path2docker(OUT_PATH)

model = YOLO(MODEL_PATH)

results = model.predict(IN_PATH,conf=conf_thr,
                      project= OUT_PATH,
                      name="inferred",
                      retina_masks=True,
                      line_width=1,
                      batch=2, 
                      save_txt =True,
                      save_conf = False,
                      device='cuda', 
                      half=True, 
                      boxes=False,
                      show_labels=True,
                      save=True,
                      exist_ok=True,
                      imgsz=[1024,768],
                      agnostic_nms=True,
                      max_det=10)