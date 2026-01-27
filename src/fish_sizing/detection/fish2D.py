import cloudpickle as pickle

import numpy as np
import cv2
import os
class Fish2D:
    def __init__(self, frame_id, color_id, fish_class,model_classes_dict,mask, bbox,
                 class_colours_dict,model_used="unknown_yolo",track_id=None, 
                 in_image_borders=False,debug_path=""):
        
        self.fish_frame = frame_id
        self.color_id = color_id  # object id (id in the frame)
        self.track_id = track_id
        self.class_name = fish_class
        
        self.bbox = bbox # [x1,y1,x2,y2]
        self.mask = mask # binary mask (img bckgrnd with mask=1 for fish pixels)
        
        self.in_image_borders = in_image_borders        
        
        self.model_used = model_used
        self.model_classes_dict = model_classes_dict
        self.class_colours_dict = class_colours_dict
        self.is_3d_complete = -1
        
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

        # Buscar contornos del objeto
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if contours:
            # Combinar contornos en uno solo
            contour = np.concatenate(contours)

            if self.debug_mode:
                cv2.drawContours(image_rgb, contours, -1, (0, 255, 0), thickness=1)
                cv2.imwrite(
                    os.path.join(self.debug_path, f"{self.fish_frame}_object{object_id}_contours_{disp_or_mask}.png"),
                    image_rgb
                )

            if contour.shape[0] > 5:
                try:
                    # Ajustar elipse sobre la envolvente convexa
                    hull = cv2.convexHull(contour).reshape(-1, 1, 2)
                    ellipse = cv2.fitEllipse(hull)
                    length = max(ellipse[1])  # Eje mayor

                    if self.debug_mode:
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

    def is_complete(self, disp_img, mask_img, debug_path="", debug_mode=True):
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
            if disp_img.shape != mask_img.shape:
                print("disp_img.shape:", disp_img.shape)
                print("mask_img.shape:", mask_img.shape)
                raise ValueError("`disp_img` y `mask_img` deben tener la misma forma.")

            # Multiplica máscara por disparidad para aislar el objeto
            disp_of_object = mask_img * disp_img

            # Cálculo de IoU
            iou = self.get_IoU_disp_object_id(mask_img, disp_of_object)

            # Visualización
            object_img_color = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)

            # Obtener longitudes de las ellipses que contienen cada mascara
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
                
                mask_and_disp = (disp_img * 255.0).astype(np.uint8) - ((mask_img // 255) * 100)
                
                cv2.imwrite(os.path.join(debug_path, f"{prefix}_disp.png"), disp_of_object)
                cv2.imwrite(os.path.join(debug_path, f"{prefix}_mask.png"), mask_img)
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

    
    def to_ros_msg(self):
        from stereo_plome.msg import FrameScene as FrameSceneMsg
        from stereo_plome.msg import Fish2D as Fish2DMsg
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
        msg = Fish2DMsg()
        msg.fish_frame = self.fish_frame
        msg.color_id = self.color_id
        msg.track_id = self.track_id 
        msg.class_name = self.class_name
        msg.in_image_borders = self.in_image_borders
        msg.model_used = self.model_used
        msg.model_classes_keys = list(self.model_classes_dict.keys())
        msg.model_classes_values = list(self.model_classes_dict.values())
        msg.class_colours_keys = list(self.class_colours_dict.keys())
        msg.class_colours_values = list(self.class_colours_dict.values())
        return msg

    @classmethod
    def from_ros_msg(cls, msg):
        from stereo_plome.msg import FrameScene as FrameSceneMsg
        from stereo_plome.msg import Fish2D as Fish2DMsg
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
        
        model_classes_dict = dict(zip(msg.model_classes_keys, msg.model_classes_values))
        class_colours_dict = dict(zip(msg.class_colours_keys, msg.class_colours_values))
        return cls(
            frame_id=msg.fish_frame,
            color_id=msg.color_id,
            track_id=msg.track_id,
            fish_class=msg.class_name,
            model_classes_dict=model_classes_dict,
            class_colours_dict=class_colours_dict,
            model_used=msg.model_used,
            in_image_borders=msg.in_image_borders
        )
        
    def __str__(self):
        return f"[Fish] {self.class_name} (ID: {self.color_id}, Track: {self.track_id})"

                
class FrameScene:
    def __init__(self, frame_name: str, object_ids_mask, fish_list, disparity_img, save_path, class_ids_img=None):
        self.frame_name = frame_name
        self.class_ids_img = class_ids_img # (Img) Each fish class is identified with a different colour in the mask
        self.object_ids_mask = object_ids_mask # (Img) Each fish object has a different color id that serves as object id
        self.fish_list = fish_list
        self.disparity_image = disparity_img
        self.debug_mode=True
        self.save_path = save_path
        
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
        cv2.imwrite(os.path.join(self.debug_path,self.frame_name + "binarized_disp_image.png"), self.disparity_image * 255)  # Save the binarized image

        objects = set(self.object_ids_mask.flatten())
        print("I found ", len(objects), " objects")
    
        
    def to_ros_msg(self):
        from stereo_plome.msg import FrameScene as FrameSceneMsg
        from stereo_plome.msg import Fish2D as Fish2DMsg
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge

        bridge = CvBridge()
        msg = FrameSceneMsg()
        msg.frame_name = self.frame_name
        msg.object_ids_mask = bridge.cv2_to_imgmsg(self.object_ids_mask, encoding="passthrough")
        msg.class_ids_img = bridge.cv2_to_imgmsg(self.class_ids_img, encoding="passthrough") if self.class_ids_img is not None else Image()
        msg.fish_list = [fish.to_ros_msg() for fish in self.fish_list]
        return msg

    @classmethod
    def from_ros_msg(cls, msg):
        from stereo_plome.msg import FrameScene as FrameSceneMsg
        from stereo_plome.msg import Fish2D as Fish2DMsg
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge

        bridge = CvBridge()
        object_ids_mask = bridge.imgmsg_to_cv2(msg.object_ids_mask, desired_encoding="passthrough")
        class_ids_img = bridge.imgmsg_to_cv2(msg.class_ids_img, desired_encoding="passthrough") if msg.class_ids_img.data else None
        fish_list = [Fish2D.from_ros_msg(fm) for fm in msg.fish_list]
        disparity_img = np.zeros_like(object_ids_mask)  # o recuperar si se transmite en el mensaje ROS
        return cls(msg.frame_name, object_ids_mask, fish_list, disparity_img, class_ids_img)

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
    





    
    