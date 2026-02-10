import os
import cv2
import numpy as np
import pickle
from ultralytics import YOLO
from termcolor import colored

from fish_sizing.detection.fish2D import Fish2D, FrameScene 

class FishDetector:
    def __init__(self, model_path, conf_thr=0.5, masks_folder="", tracker="botsort.yaml"):
        print(f"Cargando modelo YOLO: {model_path}")
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.conf_thr = conf_thr
        self.masks_folder = masks_folder
        self.tracker = tracker
                
        # Diccionarios y Colores
        self.fish_dict = self.model.names 
        self.num_classes = len(self.fish_dict.items())
        
        # Guardamos class_colours como atributo de clase (self)
        self.class_colours = {key: int(value) for key, value in zip(self.fish_dict.keys(), np.linspace(0, 255, self.num_classes + 1, dtype=int)[1:])}
            
        self.border_thickness = 10
        
    @staticmethod
    def generate_IS_bwmask(mask):
        mask_raw = mask.cpu().data.numpy().transpose(1, 2, 0)
        return np.squeeze(mask_raw)

    def process_frame(self, img, frame_id="unknown", disparity_img=None, save_debug=False, debug_path=None, save_obj=True):
        h, w = img.shape[:2]
        
        external_frame = np.ones((h, w), dtype=np.uint8)
        external_frame[self.border_thickness:-self.border_thickness, self.border_thickness:-self.border_thickness] = 0 
        
        results = self.model.track(
            img, 
            conf=self.conf_thr, 
            retina_masks=True, 
            line_width=1,
            batch=1, 
            device='cuda', 
            # project = debug_path,
            # name = self.masks_folder,
            half=True,
            agnostic_nms=True,
            show_labels=True,
            save=True, 
            augment=True,
            imgsz=1280, 
            max_det=250,
            boxes=True, 
            exist_ok=True,
            persist=True, 
            verbose=False,
            tracker=self.tracker
        )
        
        # GUARDAR LES INFERÈNCIES PEL DEBUG:
        if debug_path!=None:
           
            res = results[0]
            annotated_frame = res.plot(line_width=1, font_size=1)  # Esto te devuelve la imagen con las máscaras y cajas pintadas

            file_name = f"frame_{frame_id:05d}_inferred.jpg"  # Ej: frame_000123.jpg

            # Asegúrate de que la carpeta existe (YOLO ya no la crea por ti si save=False)
            save_folder = os.path.join(debug_path, self.masks_folder)
            os.makedirs(save_folder, exist_ok=True)

            full_save_path = os.path.join(save_folder, file_name)

            # 4. Guardar tú mismo
            cv2.imwrite(full_save_path, annotated_frame)
            

        frame_fish_list = []
        
        fish_map = {} 
        
        no_object_mask = np.full( (len(self.fish_dict.keys()), h, w), 0)
        masked_dict=dict(zip(self.fish_dict.keys(),no_object_mask))

        mask_id = np.zeros((h,w))
        track_id_mask = np.zeros((h,w))

        if(results[0].masks is not None):

            fish_masks = results[0].masks
            fish_boxes = results[0].boxes
            fish_track_ids = fish_boxes.id

            num_masks = len(fish_masks)-1
            color_ids = np.linspace(0,255,num_masks+2,dtype=int)[1:]
            
            # Go through the masks in reverse order
            for i in range(num_masks+1):

                mask = fish_masks[num_masks-i]
                bbox = fish_boxes.xyxy[num_masks-i].cpu().numpy()
                fish_cls = int(fish_boxes.cls[num_masks-i])

                mask_bw = self.generate_IS_bwmask(mask)
                mask_for_components = (mask_bw > 0).astype(np.uint8) * 255
                
                # --- NUEVO: MORPHOLOGICAL CLOSING ---
                # Lo hacemos aqui antes de calcular blobs para que una mejor las piezas
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask_for_components = cv2.morphologyEx(mask_for_components, cv2.MORPH_CLOSE, kernel)
                # Actualizamos mask_bw para reflejar el cambio (importante para solapamiento y guardado)
                mask_bw = (mask_for_components > 0).astype(np.float32)

                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_for_components, connectivity=8)
    
                if num_labels >2: 
                    # print("Num blobs in this mask: ",num_labels-1)
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    filter_mask = np.where(labels == largest_label, 1, 0).astype(np.uint8)
                    # print(colored("La mascara tiene trozos extra que estoy filtrando!!", 'yellow'))
                    mask_bw = mask_bw * filter_mask
                
                
                is_object_in_outer_frame = not(np.all((mask_bw * external_frame) == 0))
                
                if is_object_in_outer_frame:
                    # print(colored("El pez esta entrando o saliendo de la imagen", 'blue'))
                    in_image_borders = True
                else:
                    in_image_borders = False
                
                # --- NUEVO: OVERLAP CHECK ---
                current_does_overlap = False
                current_overlapping_ids = []
                
                # Comprobamos intersección con lo acumulado hasta ahora (mask_id)
                intersection = np.logical_and((mask_id > 0), (mask_bw > 0))
                intersect_area = np.count_nonzero(intersection)
                
                if intersect_area > 0:
                    current_does_overlap = True
                    # Vemos con QUÉ peces anteriores hemos chocado
                    touched_color_ids = np.unique(mask_id[intersection])
                    
                    for t_cid in touched_color_ids:
                        t_cid = int(t_cid)
                        if t_cid == 0: continue
                        
                        # Recuperar pez anterior y marcarlo también
                        if t_cid in fish_map:
                            prev_fish = fish_map[t_cid]
                            prev_fish.does_overlap = True # Marcamos el anterior
                            # Añadimos ID del actual a la lista del anterior (si tiene track)
                            track_id_curr = int(fish_track_ids[num_masks-i]) if fish_track_ids is not None else -1
                            if track_id_curr != -1:
                                prev_fish.overlapping_ids.append(track_id_curr)
                                
                            # Añadimos ID del anterior a la lista del actual
                            if prev_fish.track_id != -1:
                                current_overlapping_ids.append(prev_fish.track_id)

                    if len(current_overlapping_ids) > 0:
                        print(colored(f"⚠️ SOLAPAMIENTO DETECTADO con {current_overlapping_ids}", "magenta"))

                # Create inverted mask (Tu código original intacto)
                inverted_mask = (mask_bw*(-1)) + np.ones(mask_bw.shape)
                mask_id = mask_id*inverted_mask
                
                current_color_id = int(color_ids[num_masks-i])
                mask_id = mask_id+mask_bw*current_color_id 

                print("THIS FISH IS A: ",self.fish_dict[fish_cls],"with a confidence of : ",float(fish_boxes.conf[num_masks-i]))

                if fish_boxes[num_masks-i].is_track:
                    fish_track_id = int(fish_track_ids[num_masks-i])
                    if save_debug: # Asumimos save_debug para track img
                        track_id_mask = track_id_mask*inverted_mask
                        track_id_mask = track_id_mask+mask_bw*(255-fish_track_id*10)
                else:
                    print(colored("Im a stupid model and I'm not tracking this fish :S", 'red'))
                    fish_track_id = -1
                    pass

                for key in self.fish_dict:
                    masked_dict[key] = masked_dict[key]*inverted_mask
                    if key==fish_cls:
                        masked_dict[fish_cls] = masked_dict[fish_cls]+(mask_bw*self.class_colours[fish_cls])
                        
                current_fish = Fish2D(
                    frame_id = frame_id,
                    color_id = current_color_id,
                    model_classes_dict = self.fish_dict,
                    class_colours_dict = self.class_colours,
                    mask = mask_bw,
                    bbox = bbox,
                    fish_class = fish_cls,
                    model_used = self.model_path,
                    track_id = fish_track_id,
                    in_image_borders = in_image_borders,
                    does_overlap = current_does_overlap,
                    overlapping_ids = current_overlapping_ids,
                    debug_path = os.path.join(debug_path,"debug") if debug_path else ""
                )
                
                frame_fish_list.append(current_fish)
                # Guardamos referencia para la siguiente iteración
                fish_map[current_color_id] = current_fish
            
            # Save to disk
            final_masks = np.stack(list(masked_dict.values()))
            mask_final = np.sum(final_masks,axis=0)
            
            frame_scene = FrameScene(
                frame_name = frame_id,
                img_size = img.shape,
                fish_list = frame_fish_list,
                object_ids_mask = mask_id,
                class_ids_img = mask_final,
                disparity_map = disparity_img if disparity_img is not None else np.zeros_like(mask_id),
                save_path = os.path.join(debug_path,"debug") if debug_path else ""
            )    

            if save_debug and debug_path:
                self._save_debug_data(debug_path, frame_id, mask_final, mask_id, track_id_mask)
                
            if save_obj and debug_path:
                pkl_path = os.path.join(debug_path, self.masks_folder, str(frame_id) + "_scene.pkl")
                os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
                frame_scene.save(pkl_path)
                
        else:
             # Retorno vacío si no hay máscaras
             return False, FrameScene(frame_id, [], np.zeros((h,w)), None, None, "")

        return True, frame_scene

    def _save_debug_data(self, base_path, frame_id, class_mask, id_mask, track_mask):
        out_folder = os.path.join(base_path, self.masks_folder,"debug")
        os.makedirs(out_folder, exist_ok=True)
        cv2.imwrite(os.path.join(out_folder, f"{frame_id}_masked.png"), class_mask)
        cv2.imwrite(os.path.join(out_folder, f"{frame_id}_object_ids.png"), id_mask)
        cv2.imwrite(os.path.join(out_folder, f"{frame_id}_track_ids.png"), track_mask)
        print(colored(f"[DEBUG] Saved debug images for {frame_id}", "cyan"))