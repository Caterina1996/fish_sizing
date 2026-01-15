#!/usr/bin/env python3


from ultralytics import YOLO
import os
import cv2
import numpy as np
from natsort import natsorted
import argparse
from termcolor import colored
from fish2D import Fish2D,FrameScene



"""
This code ...
-Image with plotted inference
-Image_masked -> Plots the found objects' masks with a different colour for each class
-Image_object_ids -> Plots each object mask with a different colour
-Image_object_ids-> Plots each object with a different track id with a different colour


Binary model:
python3 /home/uib/DL_utils/ros_dl_ws/src/yolov8_inference/scripts/inference_and_plot_bw_mask.py \ 
-dp /home/uib/Escritorio/prueba/  -mp /home/uib/DATA/PEIX_PISCINA/11_41_33/ -mp /home/uib/models/Segmentation/binary_fish/nano_fold2_binary.pt

Multiclass model:
python3 /home/uib/DL_utils/ros_dl_ws/src/yolov8_inference/scripts/inference_and_plot_bw_mask.py -dp /home/uib/DATA/LIMA/Llobarros/2024_11_27/12_00_27/ -mp /home/uib/models/Segmentation/19c/UIB_OBSEA_NANO_19c.pt

python3 /home/rosuser/repo/src/DL_utils/ros_dl_ws/src/yolov8_inference/ \
    -dp /home/rosuser/repo/out/SARMIENTO/BITER_PLOME/2023/pending/1/ \
    -mp /home/rosuser/repo/dataset/models/yv11l/ylarge_dfishthings_d18_poolv2r_lantytr_nocturnes/weights/best.pt


"""

def generate_IS_bwmask(mask):
    """
    Convert the yolov8 tensor to a numpy array and transpose it  because yolov8 format 
    comes in (Num masks, H, W) and we want it in (H, W, Num masks)
    """
    mask_raw = mask.cpu().data.numpy().transpose(1, 2, 0)
    # cv2.imwrite("Maskara.jpg", mask_raw)
    return np.squeeze(mask_raw)

def save_debug_images(base_path, frame_id, masks_folder, 
                    class_mask, id_mask, track_mask=None, 
                    save_track_img=False, debug_mode=True):
    """
    Guarda las imágenes de depuración como class_mask, id_mask y track_id_mask si se desea.
    """
    if not debug_mode:
        return
    os.makedirs(os.path.join(base_path, masks_folder), exist_ok=True)

    masked_path = os.path.join(base_path, masks_folder, f"{frame_id}_masked.png")
    ids_path = os.path.join(base_path, masks_folder, f"{frame_id}_object_ids.png")

    cv2.imwrite(masked_path, class_mask)
    cv2.imwrite(ids_path, id_mask)

    if save_track_img and track_mask is not None:
        track_path = os.path.join(base_path, masks_folder, f"{frame_id}_track_ids.png")
        cv2.imwrite(track_path, track_mask)

    print(colored(f"[DEBUG] Imágenes guardadas para frame {frame_id}", "cyan"))
    
    
class FishDetector:
    def __init__(self, model_path, conf_thr=0.5, masks_folder="inferred",tracker="botsort.yaml"):
        """
        Inicializa el modelo de detección y configuraciones estáticas.
        """
        print(f"Cargando modelo YOLO: {model_path}")
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.conf_thr = conf_thr
        self.masks_folder = masks_folder
        self.tracker = tracker
                
        # Diccionarios y Colores
        self.fish_dict = self.model.names 
        self.num_classes = len(self.fish_dict.items())
        
        # Create a color_dict for each class. First interval should be background class (black)
        class_colours = {key: int(value) for key, value in zip(fish_dict.keys(), np.linspace(0, 255, num_classes + 1, dtype=int)[1:])}
            
        # To check if the fish is going out of the image:
        self.border_thickness = 10
        
    def process_frame(self, img, frame_id="unknown", disparity_img=None, save_debug=False, debug_path=None,save_obj=True):
        """
        Procesa UN solo frame (numpy array).
        
        Args:
            img: Imagen BGR (numpy array).
            frame_id: String identificador (para logs y objetos).
            save_debug: Si True, guarda las imágenes intermedias en disco.
            debug_path: Ruta base donde guardar si save_debug es True.
            save_obj: Guarda los objetos como pkl para poder leerlos luego
            
        Returns:
            frame_scene: Objeto FrameScene con toda la info.
            debug_data: Tupla (mask_final, mask_id) para visualización externa.
        """
        h, w = img.shape[:2]
        
        # 1. Frame to check if the fish is in the outer frame of the image (entering or abandoning the scene so likely to not be complete)
        external_frame = np.ones((h, w), dtype=np.uint8)
        external_frame[self.border_thickness:-self.border_thickness, self.border_thickness:-self.border_thickness] = 0 
        
        # 2. YOLO inference (Tracking)
        results = self.model.track(
            img, 
            conf=self.conf_thr, 
            retina_masks=True, 
            line_width=1,
            batch=1, 
            device='cuda', 
            half=True,
            agnostic_nms=True,
            show_labels=True,
            save = save_debug,
            project = debug_path,
            augment =True,
            name = self.masks_folder,
            exist_ok=True,
            show_labels=True,
            imgsz=1280, # TODO: maybe use 1024 given that these images are smaller
            max_det=250,
            boxes=True, 
            persist=True, 
            verbose=False,
            tracker= self.tracker
        )
        # tracker="/home/uib/DL_utils/ros_dl_ws/src/yolov8_inference/config/trackers/botsort.yaml")

 
        frame_fish_list = []
        
        # Initialize the mask image of each class to 0s (background) for this frame
        no_object_mask = np.full( (len(fish_dict.keys()), h, w), 0)
        masked_dict=dict(zip(fish_dict.keys(),no_object_mask))

        # Initialize the mask and track ids images to 0s too (no object)
        mask_id = np.zeros((h,w))
        track_id_mask = np.zeros((h,w))

        # Process inference results for this image:
        if(results[0].masks is not None):

            # get boxes and masks objects:
            fish_masks = results[0].masks
            fish_boxes = results[0].boxes
            fish_track_ids = fish_boxes.id

            # For visualization purposes we plot the object ids image
            # Each object(fish) in the image is assigned with a different colour 
            num_masks = len(fish_masks)-1
            # We need an additional color for background
            color_ids = np.linspace(0,255,num_masks+2,dtype=int)[1:]
            
            # Go through the masks in reverse order because the confidence list is stored tidied from lower to higher
            for i in range(num_masks+1):

                # Read mask info
                mask = fish_masks[num_masks-i]
                bbox = fish_boxes.xyxy[num_masks-i].cpu().numpy()
                fish_cls = int(fish_boxes.cls[num_masks-i])

                # Convert the yolov8 tensor to a numpy array and transpose it because yolov8 format comes in (C, H, W) and we want it in (H, W, C)
                mask_bw = generate_IS_bwmask(mask)
                mask_for_components = (mask_bw > 0).astype(np.uint8) * 255
                
                # MORPHOLOGICAL CLOSING (Suavizar dientes de sierra) --- # TODO: Repensar i chequejar que aixo no causi problemes -> esto deberia ir despues antes de is_obkect in outer frame?
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask_for_components = cv2.morphologyEx(mask_for_components, cv2.MORPH_CLOSE, kernel)
                
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_for_components, connectivity=8)
                # stats : [ left, top, width, height, area ]
    
                if num_labels >2: # El fondo ya és una label
                    print("Num blobs in this mask: ",num_labels-1)
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    filter_mask = np.where(labels == largest_label, 1, 0).astype(np.uint8)
                    print(colored("La mascara tiene trozos extra que estoy filtrando!!", 'yellow'))
                    mask_bw = mask_bw * filter_mask
                
                
                is_object_in_outer_frame = not(np.all((mask_bw * external_frame) == 0))
                
                if is_object_in_outer_frame:
                    print(colored("El pez esta entrando o saliendo de la imagen !!!! SEGURAMENTE NO ESTA ENTEROOO", 'blue'))
                    # cv2.imwrite(os.path.join(data_path,masks_inf_folder,img.split("_left")[0]+"_masked_borders.png"),(mask_bw * external_frame)*255)
                    in_image_borders = True
                
                else:
                    in_image_borders = False
                    
                current_overlapping_ids = []    
                intersection = np.logical_and((mask_id > 0), (mask_bw > 0))
                intersect_area = np.count_nonzero(intersection)
                    

                # Create inverted mask (This is to clean possible overlapping areas with previous masks)
                inverted_mask = (mask_bw*(-1)) + np.ones(mask_bw.shape)
                mask_id = mask_id*inverted_mask
                mask_id = mask_id+mask_bw*color_ids[num_masks-i] # Each mask will have a different color id in order to visualize it
                
                if intersect_area > 0:
                    area_curr = np.count_nonzero(mask_bw)
                             
                    # 2. Who is intersecting with who
                    # Extraemos los IDs de color que hay en la zona de intersección
                    touched_color_ids = np.unique(mask_id[intersection])
                    
                    for touched_cid in touched_color_ids:
                        touched_cid = int(touched_cid)
                        if touched_cid == 0: continue # Ignorar fondo
                        
                        # Recuperar el pez anterior y avisarle
                        if touched_cid in fish_map:
                            prev_fish = fish_map[touched_cid]
                            
                            # Avisamos al pez anterior (bidireccional)
                            prev_fish.does_overlap = True
                            prev_fish.overlapping_ids.append(fish_track_id) # Le decimos quién le ha tocado
                            
                            # Me apunto quién es él
                            current_overlapping_ids.append(prev_fish.track_id)
                            
                    if len(current_overlapping_ids) > 0:
                        print(colored(f"⚠️ SOLAPAMIENTO: Pez {fish_track_id} toca a {current_overlapping_ids}", "magenta"))
                

                print("THIS FISH IS A: ",fish_dict[fish_cls],"with a confidence of : ",float(fish_boxes.conf[num_masks-i]))

                # Check if the fish is being tracked
                if fish_boxes[num_masks-i].is_track:
                    fish_track_id = int(fish_track_ids[num_masks-i])
                    # print("fish track id!!!",fish_id)
                    if save_track_img:
                        track_id_mask = track_id_mask*inverted_mask
                        track_id_mask = track_id_mask+mask_bw*(255-fish_track_id*10)
                else:
                    print(colored("Im a stupid model and I'm not tracking this fish :S", 'red'))
                    fish_track_id = -1
                    pass

                for key in fish_dict:
                    #set the pixels to 0 in case they were already marked by a previous overlapping mask
                    masked_dict[key] = masked_dict[key]*inverted_mask

                    if key==fish_cls:
                        # mark the mask pixels
                        masked_dict[fish_cls] = masked_dict[fish_cls]+(mask_bw*class_colours[fish_cls])
                        
                current_fish = Fish2D(
                    frame_id = frame_id,
                    color_id = int(color_ids[num_masks-i]),
                    model_classes_dict = self.fish_dict,
                    class_colours_dict = self.class_colours,
                    mask = mask_bw,
                    bbox = bbox,
                    fish_class = fish_cls,
                    model_used = self.model_path,
                    track_id = fish_track_id,
                    in_image_borders = in_image_borders,
                    debug_path = os.path.join(debug_path,"debug"))
                
                frame_fish_list.append(current_fish)

            
            # Save to disk
            pickle_path = os.path.join(data_path, masks_inf_folder, frame_id + "_scene.pkl")
    
            final_masks = np.stack(list(masked_dict.values()))
            mask_final = np.sum(final_masks,axis=0)
            
            # Create scene wrapper
            frame_scene = FrameScene(
                frame_name = frame_id,
                fish_list = frame_fish_list,
                object_ids_mask = mask_id,
                class_ids_img = mask_final,
                disparity_img = disparity_img if disparity_img is not None else np.zeros_like(mask_id),
                save_path = os.path.join(debug_path,"debug")
                
            )    

            # Save objects and debug images:
            if save_debug and debug_path:
                
                self._save_debug_data(debug_path, frame_id, mask_final_classes, mask_id, track_id_mask)
                
                # Guardar pickle también si se desea
                os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
                frame_scene.save(pkl_path)
                frame_scene.draw_object_contours()
                
        return frame_scene, (mask_final_classes, mask_id)

    def _save_debug_data(self, base_path, frame_id, class_mask, id_mask, track_mask):
        """Método privado para guardar imágenes de debug"""
        out_folder = os.path.join(base_path, self.masks_folder)
        os.makedirs(out_folder, exist_ok=True)
        
        cv2.imwrite(os.path.join(out_folder, f"{frame_id}_masked.png"), class_mask)
        cv2.imwrite(os.path.join(out_folder, f"{frame_id}_object_ids.png"), id_mask)
        cv2.imwrite(os.path.join(out_folder, f"{frame_id}_track_ids.png"), track_mask)
        print(colored(f"[DEBUG] Saved debug images for {frame_id}", "cyan"))

