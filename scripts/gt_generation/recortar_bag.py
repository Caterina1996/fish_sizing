import rosbag
import rospy
import os

def retallar_bag_per_frames(input_bag, output_bag, camera_topic, start_frame, end_frame=None):
    """
    Retalla un fitxer .bag basant-se en el número de frame d'un tòpic de càmera.
    """
    print(f"[{input_bag}] Analitzant frames del tòpic: {camera_topic}...")
    
    start_time = None
    end_time = None
    frame_count = 0
    
    # PASSA 1: Cercar els timestamps exactes dels frames
    with rosbag.Bag(input_bag, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[camera_topic]):
            frame_count += 1
            
            if frame_count == start_frame:
                start_time = t
                print(f"   -> Frame inicial ({start_frame}) trobat a t={start_time.to_sec()}")
                
            if end_frame and frame_count == end_frame:
                end_time = t
                print(f"   -> Frame final ({end_frame}) trobat a t={end_time.to_sec()}")
                break # Ja hem trobat el final, no cal seguir llegint
                
        # Si no hem definit end_frame (o és més gran que el bag), agafem el final del bag
        if end_time is None:
            end_time = rospy.Time.from_sec(bag.get_end_time())
            print(f"   -> Frame final no assolit, tallant fins al final del bag (t={end_time.to_sec()})")

    if start_time is None:
        print("❌ Error: El frame inicial és més gran que el total de frames del bag.")
        return

    # PASSA 2: Escriure el nou bag amb TOTS els tòpics sincronitzats
    print(f"[{output_bag}] Generant nou bagfile...")
    with rosbag.Bag(input_bag, 'r') as in_bag, rosbag.Bag(output_bag, 'w') as out_bag:
        missatges_escrits = 0
        for topic, msg, t in in_bag.read_messages():
            if start_time <= t <= end_time:
                out_bag.write(topic, msg, t)
                missatges_escrits += 1
                
    print(f"✅ Fet! Nou bag guardat amb {missatges_escrits} missatges en total.\n")


# ==========================================
# EXEMPLE D'ÚS BASAT EN LES TEVES NOTES
# ==========================================
if __name__ == "__main__":
    # El tòpic del qual extreus els frames (ajusta'l al teu bag, pot ser /camera/color/image_raw, etc.)
    TOPIC_IMATGE = "/stereo_ch3/left/image_raw" 
    
    # Suposem que tens el bag: 10-21-16_0_compressed.bag
    bag_original = "/home/slimbook/bagfiles/LIMA/2025/2025_08_21/selec2/2025_08_21/10_36_30/stereo_camera_images_2025-08-21-10-37-15_1.bag"
    
    # Segons les teves notes: "Dividir en 2 a partir del f 461... Retallar del 553 al final?"
    
    # Tall 1: Del frame 26 al 460
    retallar_bag_per_frames(
        input_bag=bag_original, 
        output_bag="/home/slimbook/bagfiles/LIMA/2025/2025_08_21/retalls/10-21-16_0_part1.bag", 
        camera_topic=TOPIC_IMATGE, 
        start_frame=26, 
        end_frame=460
    )
    
    # Tall 2: Del frame 553 fins al final
    retallar_bag_per_frames(
        input_bag=bag_original, 
        output_bag="10-21-16_0_part2.bag", 
        camera_topic=TOPIC_IMATGE, 
        start_frame=553, 
        end_frame=None # None significa "fins que s'acabi el vídeo"
    )