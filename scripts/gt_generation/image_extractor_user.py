r"""
Sctipt to extract images from a single ROS bagfile with optional preprocessing (CLAHE/Color correction)
Supports both Raw and Compressed image topics, recovering original color from Bayer encodings.

By default do_preprocess is False. If a change wants to be made regarding the preprocess: 
go to the __preprocess_image_deep_sea function and modify it as you whish.
"""
import cv2
import numpy as np
from pathlib import Path
import rosbag
from cv_bridge import CvBridge
import re
from typing import List, Tuple, Optional
import argparse

class ImageExtractor():
    r"""
    Class responsible to extract frames from a specific topic in a bagfile and save them as JPGs.
    """

    def __init__(self, bag_path: str, output_folder: Path):
        self.__bag_path = Path(bag_path)
        self.__output_folder = output_folder
        self.__bridge = CvBridge()

        # PRIORITY DEFINITION (REGEX)
        # 1. Left/Port (maximum priority)
        # 2. Right/Starboard
        # 3. Generic (image_raw)
        # Note: '.*' allows ANY prefix (namespace). final ? means the expression is optional (it may exist or not)
        self.__priority_patterns =[
            re.compile(r'.*/(?:left|port)/image_(?:raw|rect)(?:/compressed)?$'),
            re.compile(r'.*/(?:right|starboard)/image_(?:raw|rect)(?:/compressed)?$'),
            re.compile(r'.*/image_(?:raw|rect)(?:/compressed)?$')
        ]
    """
    Function: extract_images

    """
    def extract_images(self, do_preprocess: bool = False) -> int:
        r"""
        PUBLIC METHOD: Main execution method. Extracts images and returns number of images extracted.
        
        Parameters
        -------------
        do_process: bool
            If we want to preprocess the images or not
        
        Returns
        -------------
        int
            The number of extracted images.

        """
        if not self.__bag_path.exists():
            print(f"[ERROR] Bagfile not found: {self.__bag_path}")
            return 0
        
        # If the output folder has images already -> We assume the work is done (SKIPPING extraction)
        if self.__output_folder.exists():
            existing_images = list(self.__output_folder.glob("*.jpg"))
            if len(existing_images) > 10:
                print(f"    [IMAGES] Found {len(existing_images)} existing images. Skipping extraction.")
                return len(existing_images)
            
        else:
            self.__output_folder.mkdir(parents=True, exist_ok=True)

        print(f"    [IMAGES] Extracting from {self.__bag_path.name} (Preprocess: {do_preprocess})...")

        count = 0

        try:
            with rosbag.Bag(str(self.__bag_path), 'r') as bag:
                # 1. Obtain all topics in the bag
                bag_topics = bag.get_type_and_topic_info()[1].keys()

                # 2. Search for best topic using Regex
                # target_topic, is_compressed = self.__find_best_topic(bag_topics)
                result = self.__find_best_topic(bag_topics)
                print(f"[DEBUG] find_best_topic returned: {result} (Type: {type(result)})")
                target_topic, is_compressed = result

                if not target_topic:
                    print(f"    [WARN] No suitable camera topic found in: {self.__bag_path.name}")
                    print(f"           Available topics: {bag_topics}") # DEBUG: Ver qué hay si falla

                print(f"    [IMAGES] Extracting from topic: {target_topic} (Preprocess: {do_preprocess})")

                # 3. Extract
                print(f"    [IMAGES] Extracting from topic: {target_topic} (Preprocess: {do_preprocess})")

                # Using this SAFE way to DEBUG:
                message_generator = bag.read_messages(topics=[target_topic])
                
                for i, bag_tuple in enumerate(message_generator):
                    try:
                        # Manual Unpacking (We will see if it fails)
                        topic, msg, t = bag_tuple 
                        
                        # DEBUG: See how to first message is like
                        if i == 0:
                            print(f"    [DEBUG MSG] Frame 0 loaded successfully.")
                            if is_compressed:
                                print(f"       - Format: {getattr(msg, 'format', 'Unknown')}")
                            else:
                                print(f"       - Encoding: {getattr(msg, 'encoding', 'Unknown')}, Size: {msg.width}x{msg.height}")

                        # ROS Image -> OpenCV Image 
                        cv_image = None

                        if is_compressed:
                            # 3.1. Obtain decoded image (may be 2D if it is Bayer)
                            # Using 'passthrough' to avoid strange automatic conversions
                            raw_image = self.__bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
                            
                            # 3.2. Recover real color looking at msg.format
                            cv_image = self.__recover_color_from_format(raw_image, msg.format)
                        
                        else:
                            # Raw - Manual Bypass Try
                            # if it is a known Bayer format, we make it
                            if 'bayer' in msg.encoding.lower() or 'mono' in msg.encoding.lower():
                                cv_image = self.__manual_bayer_to_bgr(msg)
                            else:
                                # Fallback for other formats (ex. rgb8)
                                cv_image = self.__bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

                        # 3.3. Submarine Preprocess
                        if do_preprocess:
                            cv_image = self.__preprocess_image_deep_sea(cv_image)

                        # Save image
                        img_name = f"frame_{count:06d}.jpg"
                        cv2.imwrite(str(self.__output_folder / img_name), cv_image)
                        count += 1
                    
                    except Exception as e:
                        # IMPRIMIR EL ERROR REAL
                        # Solo lo imprimimos una vez para no saturar la consola
                        if count == 0:
                            print(f"\n[CRITICAL ERROR in Frame {i}]")
                            print(f"    Tipo de error: {type(e).__name__}")
                            print(f"    Mensaje: {e}")
                            print("    Traceback:")
                            import traceback; traceback.print_exc()
                            print("-" * 30)
                            
                            # Rompemos el bucle si falla el primero, porque fallarán todos
                            break 
                        continue
                        # Sometimes there are corrupt frames, is better to not stop all the process
                        # print(f"    [WARN] Ha habido algun problema con el mensaje {msg} del topic {target_topic}")
                        # continue

            print(f"    [IMAGES] Extracted {count} images.")
            return count
        
        except Exception as e:
            print(f"[ERROR] Failed extracting images: {e}")
            return 0
        
    
    """
    Function: __find_best_topic
    
    """
    def __find_best_topic(self, available_topics: List[str]) -> Tuple[Optional[str], bool]:
        r"""
        Iterates through priority patterns and matches against available topics.

        Parameters
        -----------
        available_topics: List[str]
            List of all bag topics

        Returns
        -------------
        Tuple[Optional[str], bool]
            tuple of matching topics and if it is compressed or not
        """
        # Go through patterns in order of priority (Left -> Right -> Generic)
        for pattern in self.__priority_patterns:
            # Check each pattern with all topics in bag
            for topic in available_topics:
                if pattern.match(topic):
                    is_compressed = "compressed" in topic
                    return topic, is_compressed
                
        return None, False


    """
    Function: __recover_color_from_format
    
    """
    def __recover_color_from_format(self, image: np.ndarray, enconding_string: str) -> np.ndarray:
        r"""
        Smart DeBayering: Checks msg.format and converts 2D Bayer -> 3D BGR.

        Parameters
        ----------
        image: ndarray
            image to check/recover

        enconding_string: str
            encoding to help to recover

        Returns
        -------
        ndarray
            converted image
        """
        # If it has 3 channels, assume is BGR/RGB and return it as it is
        if len(image.shape) == 3:
            return image
        
        # If it is 2D -> DeBayering is needed
        format_str = enconding_string.lower()

        # Standard mapping from ROS to OpenCv (Bayer -> BGR)
        code = None
        if "rggb" in format_str:
            code = cv2.COLOR_BayerBG2BGR
        elif "bggr" in format_str:
            code = cv2.COLOR_BayerRG2BGR
        elif "gbrg" in format_str:
            code = cv2.COLOR_BayerGR2BGR
        elif "grbg" in format_str:
            code = cv2.COLOR_BayerGB2BGR

        if code is not None:
            try:
                # Recovering real color
                return cv2.cvtColor(image, code)
            except Exception:
                pass
        
        # Fallback: if bayer is not detected or fails. Grey -> BGR.
        # So we avoid script's crash
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    """
    Function: __preprocess_image_deep_sea
    
    """
    def __preprocess_image_deep_sea(self, image: np.ndarray, n_clahe: int = 2) -> np.ndarray:
        r"""
        Applies underwater preprocessing: Red channel adjustment + CLAHE on HSV.

        Parameters
        -----------
        image: ndarray
            image to be preprocessed
        n_clahe: int
            CLAHE parameter
        
        Returns
        -----------
        ndarray
            image preprocessed

        """
        try:
            # Work on a copy/float to avoid overflow
            img_proc = image.copy().astype(float)

            # 1. Reduce red channel values (This assumes BGR encoding)
            img_proc[:, :, 2] = img_proc[:, :, 2] * 0.6

            img_proc = img_proc.astype(np.uint8)

            # 2. Convert to HSV to separate Hue, Saturation, Value
            hsv_image = cv2.cvtColor(img_proc, cv2.COLOR_BGR2HSV)
            v_channel = hsv_image[:, :, 2]

            # 3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) just to Value,
            # to upgrade contrast without altering colors.
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            for _ in range(n_clahe):
                v_channel = clahe.apply(v_channel)

            # 4. Reconstruct
            hsv_image[:, :, 2] = v_channel
            final_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

            return final_image
        
        except Exception as e:
            print(f"[WARN] Preprocessing failed: {e}. Returning original")
            return image


    """
    Function: __manual_bayer_to_bgr
    
    """
    def __manual_bayer_to_bgr(self, msg) -> np.ndarray:
        r"""
        Converts ROS Bayer Image message to OpenCV BGR image using Numpy.
        Bypasses cv_bridge C++ extensions to avoid Conda/System library conflicts.
        """
        # 1. Convert bytes to Numpy matrix (1 channel)
        # Data come as bytes in msg.data
        dtype = np.uint8
        img_raw = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)

        # 2. Convert Bayer -> BGR using OpenCV
        # Mapping of Encodings from ROS to OpenCV
        encoding = msg.encoding.lower()
        
        if encoding == 'bayer_rggb8':
            return cv2.cvtColor(img_raw, cv2.COLOR_BayerBG2BGR)
        elif encoding == 'bayer_bggr8':
            return cv2.cvtColor(img_raw, cv2.COLOR_BayerRG2BGR)
        elif encoding == 'bayer_gbrg8':
            return cv2.cvtColor(img_raw, cv2.COLOR_BayerGR2BGR) # <--- TU CASO
        elif encoding == 'bayer_grbg8':
            return cv2.cvtColor(img_raw, cv2.COLOR_BayerGB2BGR)
        elif encoding == 'mono8':
            return cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
        
        # If it is not standard Bayer, we try to return it as it is (maybe is BGR already)
        # Or we return an Errir if it is a strange format.
        return img_raw


if __name__ == "__main__":
    # INITIALIZATIONS: --------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Extract images from bagfile")
    parser.add_argument("--bagfile_path", "-bp", type=str, help="path where the bagfile is found",default="/home/azken/caterina/bagfiles/OBSEA_09_2024/selec_TO_PLOME/2024_09_24/10_18_24/stereo_camera_images_2024-09-24-10-18-24_0.bag")
    parser.add_argument("--output_folder", "-out",type=str, help="folder where the extracted images are stored", default="/home/azken/caterina/DATA/OBSEA_09_2024/selec_TO_PLOME/2024_09_24/10_18_24/0/processed_images/")
    parser.add_argument("--preprocess",type=bool, help="preprocess images before storing them or not", default=False)

    args = parser.parse_args()

    # Instanciamos y ejecutamos
    extractor = ImageExtractor(args.bagfile_path, Path(args.output_folder))
    extractor.extract_images(do_preprocess=args.preprocess)