
from termcolor import cprint


# ==========================================
# PROCESSING PIPELINES
# ==========================================

# Default del convert_custom_linear for context
# convert_to_custom_grayscale(self,w_g=0.7,w_b=0.3,w_r=0.0)

PROCESSING_PIPELINES = {
    "raw": [],
    "basic": [
        ("match_brightness_linear", {"reference": "left"}, False),
        ("convert_to_custom_grayscale", {"w_g": 0.65, "w_b": 0.25, "w_r": 0.1}, False),
        ("apply_clahe", {"clip_limit": 2.5, "grid_size": (8, 8)}, False)      
    ],
    "sharp_stereo": [
        ("match_brightness_linear", {"reference": "left"}, False),
        ("apply_gamma",             {"gamma": 1.2}, True), 
        ("apply_bilateral",         {"d": 7, "sigma_color": 50, "sigma_space": 50}, True), 
        ("apply_sharpen",           {"alpha": 1.7}, True), 
        ("convert_to_custom_grayscale", {}, False),
        ("match_histograms",        {"reference": "left"}, True) 
    ],
    "clean_edges": [
        ("apply_bilateral",         {"d": 9, "sigma_color": 75, "sigma_space": 75}, True), 
        ("convert_to_custom_grayscale", {}, False),
        ("apply_clahe",             {"clip_limit": 1.5, "grid_size": (8, 8)}, True) 
    ],
    "dehazing": [
        ("apply_dehaze",            {"omega": 0.85}, True), 
        ("match_brightness_linear", {"reference": "left"}, False),
        ("convert_to_custom_grayscale", {}, False),
        ("apply_clahe",             {"clip_limit": 2.0, "grid_size": (8, 8)}, True),
        ("match_histograms",        {"reference": "left"}, True)
    ]
}

# ==========================================
# DOCKER PATH MAPPINGS
# ==========================================
USE_DOCKER = True

PATH_MAPPINGS = {
    "/home/slimbook/bagfiles": "/home/rosuser/dataset/bagfiles",
    "/home/slimbook/fish_sizing/out": "/home/rosuser/repo/out",
    "/home/slimbook/models": "/home/rosuser/dataset/models/",
    "/home/slimbook/fish_sizing/config": "/home/rosuser/repo/config/",
    "/media/slimbook/easystore": "/home/rosuser/easystore",
    "/media/slimbook/easystore1": "/home/rosuser/easystore1",
    "/media/slimbook/easystore2": "/home/rosuser/easystore2"
}


def transform_path2docker(path: str,use_docker=USE_DOCKER) -> str:
    """Transform a path from local computer to docker structure."""
    
    if not use_docker or path is None:
        return path

    for host_path, docker_path in PATH_MAPPINGS.items():
        if host_path in path:
            new_path = path.replace(host_path, docker_path)
            cprint(f"🔄 Path mapped: {path} \n   -> {new_path}", "yellow")
            return new_path    
    return path