# src/fish_sizing/__init__.py

# Importamos las clases desde sus archivos individuales
# El punto (.) indica importación relativa dentro del paquete

# 1. Modelos Base
from .fish2d import Fish2D, FrameScene
from .fish3d import Fish3D

# 2. Lógica de Detección
from .detector import FishDetector

# 3. Herramientas de Análisis
from .analysis import Bagfile_fauna

# 4. Configuración (si decides crear el archivo config.py)
# from .config import StereoConfig

# Definimos qué se exporta cuando alguien hace 'from fish_sizing import *'
__all__ = [
    "Fish2D",
    "FrameScene",
    "Fish3D",
    "FishDetector",
    "Bagfile_fauna"
]

__version__ = "0.1.0"

