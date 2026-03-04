import cv2
import os
from termcolor import cprint

PATH_MAPPINGS = {
    "/home/slimbook/bagfiles": "/home/rosuser/dataset/bagfiles",
    "/home/slimbook/fish_sizing/out": "/home/rosuser/repo/out",
    "/home/slimbook/models": "/home/rosuser/dataset/models/",
    "/home/slimbook/fish_sizing/config" :"/home/rosuser/repo/config/",
    "/media/slimbook/easystore": "/home/rosuser/easystore"
}

USE_DOCKER = True

def frames_to_video(input_folder, output_video, fps):
    # Obtener lista de imágenes y ordenarlas
    images = [img for img in os.listdir(input_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort() # Crucial para que el video tenga sentido cronológico

    if not images:
        print("No se encontraron imágenes en la carpeta.")
        return

    # Leer la primera imagen para obtener dimensiones
    frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, layers = frame.shape

    # Definir el codec y crear el objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        path = os.path.join(input_folder, image)
        frame = cv2.imread(path)
        video.write(frame)

    video.release()
    print(f"Video guardado exitosamente como: {output_video}")

# Configuración
folder = '/media/slimbook/easystore/DATA/SARMIENTO/07_46_28/inference_finetuning_ylarge_fold2_binary/inferred/'
video_name = '/media/slimbook/easystore/DATA/SARMIENTO/07_46_28/plome_exemple.mp4'
frames_per_second = 25

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
    folder = transform_path2docker(folder)
    video_name =transform_path2docker(video_name)


frames_to_video(folder, video_name, frames_per_second)