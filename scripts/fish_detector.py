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
This code performs inference on the left images inside a folder of extracted images.
For each image the following files are generated if desired:

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


USE yolov8 env for consistency with the numpy versions of pickle with other scripts

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


def main():
    
    # INITIALIZATIONS: --------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", "-dp", type=str, help="Data path where images for inference are",default="//media/uib/easystore/CATERINA/peixos_3D_experiments_piscina/codi_reordenat/Escenari_3_peixos_vermell/PEIX_MORT_PISCINA/2024_11_28/test/")
    parser.add_argument("--model_path", "-mp",type=str, help="path_to_model_pt", default="/home/uib/models/Segmentation/pool/25ckpt+POOL_y11_large/last.pt")
    parser.add_argument("--masks_folder", "-mf",type=str, help="masks_folder",default="inferred")
    parser.add_argument("--confidence_threshold", "-thr",type=str, help="confidence threshold for inference",default=0.5)
    parser.add_argument("--save_dict", "-sd", type=bool, default=True)
    parser.add_argument("--debug_mode", "-debug", type=bool, default=False,help="if active it stores all debug images (class_mask, id_mask y track_id_mask)")
    args = parser.parse_args()

    # Accessing arguments
    data_path = args.data_path
    model_path = args.model_path
    masks_inf_folder = args.masks_folder
    conf_thr = args.confidence_threshold
    
    save_track_img = False
    debug_mode = False
    
    disp_thr = 8 #8? this used to be one! think about it! -> disp values seem to start at 4? Never 

    
    model = YOLO(model_path)
    fish_dict = model.names 
    num_classes = len(fish_dict.items())

    # To check if the fish is going out of the image:
    external_frame =  np.ones((768, 1024), dtype=np.uint8)
    border_thickness = 10  # Grosor del marco
    external_frame [border_thickness:-border_thickness, border_thickness:-border_thickness] = 0  # Borde superior

    # Create a color_dict for each class. First interval should be background class (black)
    class_colours = {key: int(value) for key, value in zip(fish_dict.keys(), np.linspace(0, 255, num_classes + 1, dtype=int)[1:])}
    

    print("Model classes dict: ",fish_dict)
    print("Assigned class colours: ",class_colours)
    print("Number of classes is: ",num_classes)

    ## PROCESS IMAGES ------------------------------------------------------------------------
    for img in natsorted(os.listdir(data_path)):
        
        frame_fish_list = []

        #Inference
        img_path = os.path.join(data_path,img)
        
        #Perform inference just on the left image
        if (os.path.isfile(img_path)==True) and ("left" in img and "mask" not in img and "yaml" not in img and "infer" not in img and "json" not in img):

            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("Running inference on ", img_path)
            
            frame_id = img.split("_left")[0]

            
            # disparity_img_path = os.path.join(data_path,frame_id+"_disparity.png")
            # disp_img = cv2.imread(disparity_img_path)
            # disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2GRAY)
            # # Binarize disp	
            # _, disp_img_thr = cv2.threshold(disp_img, disp_thr, 1, 0)

            # If track wants to be used (recommended):
            results = model.track(img_path,conf=conf_thr,project=data_path,name=masks_inf_folder,retina_masks=True,line_width=1,
                                batch=4, device='cuda', half=True, boxes=True,show_labels=True,save=True,exist_ok=True,imgsz=[1024,768],
                                agnostic_nms=True,max_det=250,persist=True)
                                # tracker="/home/uib/DL_utils/ros_dl_ws/src/yolov8_inference/config/trackers/botsort.yaml")
            
            h, w, _ = results[0].orig_img.shape

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
                    fish_cls = int(fish_boxes.cls[num_masks-i])

                    # Convert the yolov8 tensor to a numpy array and transpose it because yolov8 format comes in (C, H, W) and we want it in (H, W, C)
                    mask_bw = generate_IS_bwmask(mask)
                    mask_for_components = (mask_bw > 0).astype(np.uint8) * 255
                    
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
                        
    
                    # Create inverted mask (This is to clean possible overlapping areas with previous masks)
                    inverted_mask = (mask_bw*(-1)) + np.ones(mask_bw.shape)
                    mask_id = mask_id*inverted_mask
                    mask_id = mask_id+mask_bw*color_ids[num_masks-i] # Each mask will have a different color id in order to visualize it

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
                        model_classes_dict = fish_dict,
                        class_colours_dict = class_colours,
                        fish_class = fish_cls,
                        model_used = model_path,
                        track_id = fish_track_id,
                        in_image_borders = in_image_borders,
                        debug_path = os.path.join(data_path,"debug")
                    
                        )
                    
                    current_fish.is_complete(disp_img_thr,mask_for_components,debug_path=os.path.join(data_path,"debug"),debug_mode=True)
                    
                    
            
                    frame_fish_list.append(current_fish)
                        
            # Save to disk
            pickle_path = os.path.join(data_path, masks_inf_folder, img.split("_left")[0] + "_scene.pkl")
            
            final_masks = np.stack(list(masked_dict.values()))
            mask_final = np.sum(final_masks,axis=0)
            
            # Create scene wrapper
            frame_scene = FrameScene(
                frame_name = img.split("_left")[0],
                fish_list = frame_fish_list,
                object_ids_mask = mask_id,
                class_ids_img = mask_final,
                disparity_img = disp_img_thr,
                save_path = os.path.join(data_path,"debug")
                
            )
            
            frame_scene.save(pickle_path)
            frame_scene.draw_object_contours()
            
            save_debug_images(data_path, img.split("_left")[0], masks_inf_folder, 
                            mask_final, mask_id, track_mask=track_id_mask, 
                            save_track_img=save_track_img, debug_mode=debug_mode)

        else:
            if "left" in img_path:
                print("NOT A LEFT IMAGE FILE!!!!: ",img_path)


if __name__ == "__main__":
    main()