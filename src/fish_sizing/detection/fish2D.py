import cloudpickle as pickle

import numpy as np
import cv2
import os
class Fish2D:
    def __init__(self, frame_id, color_id, fish_class,model_classes_dict,mask, bbox,
                 class_colours_dict, does_overlap , overlapping_ids,
                 model_used="unknown_yolo",track_id=None, 
                 aspect_ratio=-1,
                 in_image_borders=False,       
                 debug_path=""):
        
        self.fish_frame = frame_id
        self.color_id = color_id  # object id (id in the frame)
        self.track_id = track_id
        self.class_name = fish_class
        
        self.bbox = bbox # [x1,y1,x2,y2]
        self.mask = mask # binary mask (img bckgrnd with mask=1 for fish pixels)
        
        self.in_image_borders = in_image_borders      
          
        self.does_overlap = does_overlap        
        self.overlapping_ids = overlapping_ids        
        
        self.model_used = model_used
        self.model_classes_dict = model_classes_dict
        self.class_colours_dict = class_colours_dict
        self.is_3d_complete = -1
        self.aspect_ratio = aspect_ratio
        
        self.debug_mode = True
        self.debug_path = debug_path

    
    @staticmethod
    def get_IoU_disp_object_id(obj_image,disp_of_object):
        return np.count_nonzero(disp_of_object) / np.count_nonzero(obj_image)
    
    def find_mask_length(self, image, object_id, disp_or_mask="disp"):
        """
        Encuentra la longitud del eje mayor de la elipse ajustada al contorno de un objeto.

        Args:
            image (np.ndarray): Imagen binaria (máscara del objeto o disparidad correspondiente a la zona segmentada).
            object_id (int): ID del objeto, utilizado para depuración.
            disp_or_mask (str): Etiqueta para distinguir el tipo de imagen ("disp" o "mask").

        Returns:
            tuple: (longitud del eje mayor de la elipse, elipse ajustada o None)
        """
        length = -1
        ellipse = None

        # --- FIX: Conversión obligatoria a UINT8 para findContours ---
        # Si la imagen es float (disparidad), la convertimos.
        # Si los valores son 0-1 (mascara), escalamos a 0-255.
        image_u8 = image.copy()
        if image.dtype != np.uint8:
             # Si es disparidad float, binarizamos lo que sea mayor que 0
             image_u8 = (image > 0).astype(np.uint8) * 255
        elif image.max() <= 1:
             # Si es mascara binaria 0-1, pasar a 0-255
             image_u8 = (image * 255).astype(np.uint8)

        # Buscar contornos del objeto (usando la imagen corregida image_u8)
        contours, _ = cv2.findContours(image_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Para debug usamos una copia a color
        image_rgb = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2RGB)

        if contours:
            # Combinar contornos en uno solo
            contour = np.concatenate(contours)

            if self.debug_mode and self.debug_path: # Check extra de path
                cv2.drawContours(image_rgb, contours, -1, (0, 255, 0), thickness=1)
                cv2.imwrite(
                    os.path.join(self.debug_path, f"{self.fish_frame}_object{object_id}_contours_{disp_or_mask}.png"),
                    image_rgb
                )

            if contour.shape[0] > 5:
                try:
                    # Ajustar elipse sobre la envolvente convexa
                    hull = cv2.convexHull(contour).reshape(-1, 1, 2)
                    # Necesitamos al menos 5 puntos para fitEllipse
                    if hull.shape[0] >= 5:
                        ellipse = cv2.fitEllipse(hull)
                        length = max(ellipse[1])  # Eje mayor

                        if self.debug_mode and self.debug_path:
                            cv2.polylines(image_rgb, [hull], isClosed=True, color=(255, 0, 0), thickness=2)
                            cv2.ellipse(image_rgb, ellipse, (0, 0, 255), 2)
                            cv2.imwrite(
                                os.path.join(self.debug_path, f"{self.fish_frame}_object{object_id}_ellipse_{disp_or_mask}.png"),
                                image_rgb
                            )
                except Exception as e:
                    print(f"[WARN] Error al ajustar la elipse para el objeto {object_id}: {e}")
            else:

                print(f"[WARN] Contorno demasiado pequeño para el objeto {object_id} (puntos: {len(contour)})")
        else:
            print(f"[WARN] No se encontraron contornos para el objeto {object_id}")

        return length, ellipse

    def is_complete(self, disp_img, debug_path="", debug_mode=True):
        """
        Evalúa si un pez detectado está completo en base a su máscara y la imagen de disparidad.

        Se calcula el IoU entre la máscara del pez y su imagen de disparidad. Luego se estiman 
        las longitudes de ambas formas para evaluar si el objeto está bien definido en 3D. 
        Se generan imágenes de depuración con contornos y elipses si se activa `debug_mode`.

        Args:
            disp_img (np.ndarray): Imagen de disparidad (valores flotantes o enteros).
            
            debug_path (str): Carpeta donde se guardarán las imágenes de depuración.
            debug_mode (bool): Si está activado, se guardan imágenes y se imprimen métricas.

        Raises:
            IOError: Si ocurre un error al guardar las imágenes de depuración.
        """
        
        # mask_img (np.ndarray): Máscara binaria del pez (uint8).
        mask_img = self.mask
        
        try:
            # Comprobación de entrada
            if disp_img.shape[:2] != mask_img.shape[:2]: # Miramos solo H,W por si acaso disp tiene canales
                print("disp_img.shape:", disp_img.shape)
                print("mask_img.shape:", mask_img.shape)
                raise ValueError("`disp_img` y `mask_img` deben tener la misma forma.")

            # Multiplica máscara por disparidad para aislar el objeto
            # disp_img suele ser float, asi que disp_of_object será float
            disp_of_object = mask_img * disp_img

            # Cálculo de IoU
            iou = self.get_IoU_disp_object_id(mask_img, disp_of_object)

            # Visualización
            # Conversión segura para visualización
            object_img_color = cv2.cvtColor((mask_img*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

            # Obtener longitudes de las ellipses que contienen cada mascara
            # find_mask_length ahora maneja la conversion float->uint8 internamente
            length_disp, ellipse_disp = self.find_mask_length(disp_of_object, self.color_id, "DISPARITY")
            length_mask, ellipse_mask = self.find_mask_length(mask_img, self.color_id, "MASK")

            # Dibujo de elipses si están disponibles
            if ellipse_mask is not None:
                cv2.ellipse(object_img_color, ellipse_mask, (255, 0, 0), 2)  # Azul

            if ellipse_disp is not None:
                cv2.ellipse(object_img_color, ellipse_disp, (0, 255, 0), 2)  # Verde

            if debug_mode and debug_path:
                os.makedirs(debug_path, exist_ok=True)
                prefix = f"{self.fish_frame}_object_{self.color_id}"
                
                # Normalizamos disp para verla (porque en float los valores son pequeños para 0-255 o grandes)
                disp_vis = cv2.normalize(disp_of_object, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                mask_and_disp = ((disp_img > 1 )* 100.0).astype(np.uint8) + ((mask_img>0) * 100)

                cv2.imwrite(os.path.join(debug_path, f"{prefix}_disp.png"), disp_vis)
                cv2.imwrite(os.path.join(debug_path, f"{prefix}_mask.png"), (mask_img*255).astype(np.uint8))
                cv2.imwrite(os.path.join(debug_path, f"{prefix}_mask_and_disp.png"), mask_and_disp)

                cv2.imwrite(os.path.join(debug_path, f"{prefix}_ellipses.png"), object_img_color)
                
                print(f"IoU for {prefix}: {iou:.3f}")
                print(f"Length disp: {length_disp}, Length mask: {length_mask}")

            # Evaluación de si tengo info 3D para mi mascara 2D
            if length_mask == 0:
                self.is_3d_complete = -1
                print("❌ La mascara esta mal/ ha habido un error con las mascaras!!")
                return

            if (length_disp / length_mask > 0.8) and (iou > 0.75):
                self.is_3d_complete = 1
                if debug_mode:
                    print("✔️ 3D del pez completo.")
            else:
                self.is_3d_complete = 0
                if debug_mode:
                    print("❌ 3D del pez incompleto.")

        except Exception as e:
            print(f"❗ Error en is_complete: {e}")
            self.is_3d_complete = -1

    def __str__(self):
        return f"[Fish] {self.class_name} (ID: {self.color_id}, Track: {self.track_id})"

                
class FrameScene:
    def __init__(self, frame_name: str,img_size, object_ids_mask, fish_list, disparity_map, save_path, class_ids_img=None):
        self.frame_name = frame_name
        self.img_size = img_size
        self.class_ids_img = class_ids_img # (Img) Each fish class is identified with a different colour in the mask
        self.object_ids_mask = object_ids_mask # (Img) Each fish object has a different color id that serves as object id
        self.fish_list = fish_list
        self.disparity_map = disparity_map
        self.debug_mode=True
        self.save_path = save_path
        self.scene_points_3d = None
        
        if self.debug_mode:
            self.debug_path=os.path.join(self.save_path,"debug")
    
    def get_number_of_fish(self):
        """Busca un pez en la lista por su ID de color (object_id)"""
        if self.fish_list is not None:
            return len(self.fish_list)
        else:
            return 0
    
    def get_fish_from_scene(self,object_color_id):
        """Busca un pez en la lista por su ID de color (object_id)"""
        for fish in self.fish_list:
            if fish.color_id == object_color_id:
                return fish
        return None
    
    def get_fish_colors_from_scene(self):
        colors=[]
        for fish in self.fish_list:
            colors.append(fish.color_id)
        return colors

    def save(self, output_path = None):
        if output_path == None:
            output_path = self.save_path
        with open(output_path, "wb") as f:
            pickle.dump(self, f)
        print("✅ FrameScene saved to:", output_path)
        
    def draw_object_contours(self):

        print("DRAWING CONTOURNS FOR DEBUG PROPOSES")
        # _, thr_ids = cv2.threshold(self.class_ids_img, 1, 255, 0)
        _, thr_ids = cv2.threshold(self.class_ids_img.astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
        # thr_ids = (thr_ids > 0).astype(np.uint8) * 255
        thr_ids = thr_ids.astype(np.uint8)
        contours_ids, _ = cv2.findContours(thr_ids, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        ids_mat_col = cv2.cvtColor(self.class_ids_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # print("Type of contours1:", type(contours_ids))  # Should be <class 'list'>
        # print("Length of contours1:", len(contours_ids))  # Should be > 0 if contours are found

        ids_mat_col = cv2.drawContours(ids_mat_col, contours_ids, -1, (0,255,0), 3) # -1 per dibuixar tots els contorns  
        cv2.imwrite(os.path.join(self.debug_path,str(self.frame_name)+"_all_contours_ids_mask.png"), ids_mat_col)
        print("All countours plotted and saved in: ", os.path.join(self.debug_path,str(self.frame_name)+"_all_contours_ids_mask.png"))
        cv2.imwrite(os.path.join(self.debug_path,self.frame_name + "binarized_disp_image.png"), self.disparity_map * 255)  # Save the binarized image

        objects = set(self.object_ids_mask.flatten())
        print("I found ", len(objects), " objects")
    
    
    def __str__(self):
        fish_strings = "\n".join(str(fish) for fish in self.fish_list)
        return f"[Scene] {self.frame_name} with {len(self.fish_list)} fish:\n{fish_strings}"
                    
                    
    def get_detection_counts(self):
        """Return per-class counts and total detections for this frame."""
        # Initialize counts for each class
        counts = dict.fromkeys([self.fish_list[0].model_classes_dict[c] 
                                for c in self.fish_list[0].model_classes_dict] if self.fish_list else [], 0)
        
        frame_info = {"FrameId": self.frame_name, "Num_detections": 0, **counts}

        for fish in self.fish_list:
            cls_name = fish.model_classes_dict.get(fish.class_name, fish.class_name)
            frame_info[cls_name] += 1
            frame_info["Num_detections"] += 1

        return frame_info
    
    def get_optimization_strips(self, min_margin=32, force_full_threshold=0.80):
        """
        Calcula las franjas horizontales óptimas para las
        que calcularemos disparidad basándose en los peces de esta escena.
        """
        img_h = self.img_size[0]
        bboxes = [f.bbox for f in self.fish_list]
        
        if not bboxes:
            return []

        # 1. Extraer coordenadas verticales (y1, y2) con margen
        intervals = []
        for box in bboxes:
            y1, y2 = int(box[1]), int(box[3])
            y1 = max(0, y1 - min_margin)
            y2 = min(img_h, y2 + min_margin)
            intervals.append((y1, y2))

        # 2. Ordenar y Fusionar (Merge intervals)
        intervals.sort(key=lambda x: x[0])
        
        merged = []
        if intervals:
            curr_y1, curr_y2 = intervals[0]
            for next_y1, next_y2 in intervals[1:]:
                if next_y1 < curr_y2: # Solapamiento -> Extender
                    curr_y2 = max(curr_y2, next_y2)
                else: # Nuevo intervalo
                    merged.append((curr_y1, curr_y2))
                    curr_y1, curr_y2 = next_y1, next_y2
            merged.append((curr_y1, curr_y2))

        # 3. Decisión: ¿Ratio de ocupación?
        total_height = sum(y2 - y1 for y1, y2 in merged)
        coverage_ratio = total_height / float(img_h)

        if coverage_ratio > force_full_threshold:
            # print(f"   -> Modo FULL (Ocupación {coverage_ratio:.1%})")
            return [(0, img_h)]
        
        return merged
        
        
    def __str__(self):
        fish_strings = "\n".join(str(fish) for fish in self.fish_list)
        return f"[Scene] {self.frame_name} with {len(self.fish_list)} fish:\n{fish_strings}"





    
    