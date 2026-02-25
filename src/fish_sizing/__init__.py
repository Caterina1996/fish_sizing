# src/fish_sizing/__init__.py

# --- 1. Detección y Modelos 2D (Carpeta 'detection') ---
# El punto (.) significa "desde este paquete".
# Luego bajamos a la carpeta .detection y buscamos el archivo .fish2D
# from .detection.fish2D import Fish2D, FrameScene
# from .detection.fish_detector import FishDetector

# # --- 2. Medición 3D (Carpeta 'measurement') ---
# from .measurement.fish3D import Fish3D
# # Si necesitas la función de medir:
# # from .measurement.measure_fish_new import measure_fish

# # --- 3. Análisis (Carpeta 'analysis') ---
# from .analysis.bagfile_fauna import Bagfile_fauna

# --- 4. Utilidades (Carpeta 'utils') ---
# from .img_processor.image_processing import ImageProcessor
# # OJO: En tu tree pone 'bag_procesor.py' (con una sola 's'), asegúrate de escribirlo igual
# from .bag_tools.bag_processor import BagProcessor 

from .utils import tools

# # Definimos qué se exporta al hacer 'from fish_sizing import *'
# __all__ = [
#     "Fish2D",
#     "FrameScene",
#     "FishDetector",
#     "Fish3D",
#     "Bagfile_fauna",
#     "ImageProcessor",
#     "BagProcessor"
# ]

__version__ = "0.1.0"