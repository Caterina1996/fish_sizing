import rosbag
import pandas as pd
from pathlib import Path
import re

# ==========================================
# GROUND TRUTH DICTIONARY
# ==========================================
MULTIPLE_FISH_GT = {
    "2024_11_28": {
        '13-14-43_0': {1: "S_scombrus", 21: "B_boops", 24: "S_scombrus", 29: "B_boops", 37: "S_scombrus", 45: "S_scombrus", 58: "B_boops", 62: "B_boops", 75: "B_boops", 77: "B_boops", 81: "B_boops", 82: "B_boops"},
        '13-16-54_0': {1: "S_scombrus", 12: "B_boops", 17: "B_boops", 29: "B_boops", 41: "S_scombrus", 49: "S_scombrus", 53: "S_scombrus", 59: "B_boops", 67: "B_boops", 68: "B_boops", 70: "S_scombrus", 71: "B_boops"},
        '13-27-52_0': {1: "S_scombrus", 2: "B_boops", 3: "B_boops", 5: "B_boops", 6: "B_boops", 8: "B_boops", 11: "S_scombrus", 12: "B_boops", 13: "B_boops", 33: "S_scombrus", 35: "error", 37: "S_scombrus"},
        '13-32-16_0': {1: "B_boops", 4: "S_scombrus", 6: "S_scombrus", 17: "B_boops", 25: "S_scombrus", 28: "S_scombrus", 30: "S_scombrus", 44: "S_scombrus", 52: "S_scombrus", 53: "S_scombrus", 67: "S_scombrus"},
        '13-34-27_0': {1: "S_scombrus", 5: "S_scombrus", 8: "S_scombrus", 16: "S_scombrus", 21: "S_scombrus", 23: "B_boops", 28: "S_scombrus", 30: "error", 31: "S_scombrus"},
        '13-36-49_0': {1: "B_boops", 2: "S_scombrus", 3: "B_boops", 6: "B_boops", 10: "B_boops", 11: "B_boops", 19: "B_boops", 20: "S_scombrus", 24: "B_boops", 30: "S_scombrus", 33: "B_boops", 40: "error", 41: "error", 42: "error"},
        '13-39-00_0': {1: "error", 2: "S_scombrus", 8: "S_scombrus", 11: "S_scombrus", 12: "S_scombrus", 23: "error"},
        '13-41-12_0': {2: "B_boops", 3: "S_scombrus", 11: "S_scombrus"},
        '13-42-45_0': {1: "S_scombrus", 2: "B_boops", 5: "S_scombrus", 17: "S_scombrus", 26: "S_scombrus"},
        '13-48-24_0': {1: "S_scombrus", 4: "B_boops", 5: "B_boops", 10: "B_boops"},
        '13-49-20_0': {1: "S_scombrus", 2: "B_boops", 9: "B_boops", 13: "S_scombrus", 16: "B_boops", 25: "S_scombrus", 29: "B_boops", 32: "B_boops", 34: "B_boops", 36: "B_boops", 41: "B_boops", 44: "fish_1", 56: "fish_1", 58: "B_boops", 60: "fish_1", 61: "B_boops", 64: "B_boops", 69: "fish_1", 72: "fish_1", 73: "S_scombrus", 75: "fish_1", 76: "fish_1", 77: "fish_1"}
    },
    "2025_05_08": {
        '11-28-42_0': {1: "M_poutassou", 2: "D_labrax_2", 5: "M_poutassou"},
        '11-29-16_1': {1: "D_labrax_2", 2: "M_poutassou", 6: "M_poutassou", 8: "M_poutassou"},
        '11-29-49_2': {1: "M_poutassou", 2: "D_labrax_2", 6: "M_poutassou", 7: "M_poutassou"},
        '11-30-34_0': {1: "M_poutassou", 6: "D_labrax_2"},
        '11-31-08_1': {1: "D_labrax_2", 2: "M_poutassou", 3: "M_poutassou", 8: "M_poutassou"},
        '11-33-23_0': {1: "D_labrax_2", 2: "M_poutassou", 10: "M_poutassou", 18: "D_labrax_2", 19: "M_poutassou", 20: "M_poutassou"},
        '11-33-52_1': {1: "D_labrax_2", 2: "M_poutassou", 4: "D_labrax_2"},
        '11-36-36_0': {1: "D_labrax_2", 2: "M_poutassou", 8: "D_labrax_2", 13: "M_poutassou", 23: "M_poutassou", 27: "M_poutassou", 30: "D_labrax_2"}, 
        '11-37-20_1': {1: "error", 2: "D_labrax_2", 8: "M_poutassou", 16: "D_labrax_2", 18: "M_poutassou", 25: "M_poutassou", 33: "M_poutassou", 34: "M_poutassou"},
        '11-41-45_0': {1: "D_labrax_2", 8: "D_labrax_2"},
        '11-42-15_1': {1: "D_labrax_2"},
        '11-44-48_0': {1: "D_labrax_2", 3: "M_poutassou", 6: "M_poutassou", 7: "M_poutassou", 10: "error"},
        '11-48-15_0': {1: "D_labrax_2", 2: "M_poutassou", 4: "M_poutassou_tape", 6: "M_poutassou_tape", 13: "M_poutassou_tape", 14: "D_labrax_2", 16: "M_poutassou", 18: "M_poutassou_tape", 19: "M_poutassou_tape", 22: "M_poutassou_tape"},
        '11-49-52_0': {1: "D_labrax_2", 3: "M_poutassou", 4: "M_poutassou_tape", 7: "M_poutassou_tape", 10: "D_labrax_2", 13: "M_poutassou", 14: "M_poutassou_tape"},
        '11-57-17_0': {1: "D_labrax_2", 2: "M_poutassou_tape", 4: "M_poutassou"},
        '11-57-59_0': {1: "M_poutassou", 2: "M_poutassou", 4: "M_poutassou_tape", 9: "D_labrax_2", 10: "D_labrax_2", 16: "M_poutassou", 20: "M_poutassou_tape", 24: "D_labrax_2"},
        '11-59-57_0': {1: "M_poutassou", 2: "M_poutassou_tape", 8: "D_labrax_2", 14: "D_labrax_2", 20: "M_poutassou", 24: "D_labrax_2"},
        '12-00-39_0': {1: "M_poutassou_tape", 2: "D_labrax_2", 3: "M_poutassou", 14: "M_poutassou_tape", 17: "M_poutassou", 29: "M_poutassou", 34: "M_poutassou", 36: "M_poutassou", 39: "M_poutassou"},
        '12-01-20_0': {1: "M_poutassou", 2: "M_poutassou_tape", 3: "D_labrax_2", 4: "M_poutassou", 5: "D_labrax_2", 9: "M_poutassou", 13: "M_poutassou_tape", 26: "D_labrax_2", 29: "M_poutassou_tape", 33: "M_poutassou_tape"},
        '12-02-02_0': {1: "M_poutassou_tape", 2: "M_poutassou", 3: "D_labrax_2", 9: "M_poutassou", 11: "M_poutassou_tape", 13: "M_poutassou_tape", 16: "M_poutassou"},
        '12-02-44_0': {1: "D_labrax_2", 2: "M_poutassou", 3: "M_poutassou_tape", 11: "M_poutassou_tape", 13: "M_poutassou"}
    },
    "2025_08_21": {
        "10-01-53_0_compressed-r201-end": {-1: "None", 1: "D_labrax_red", 3: "D_labrax_no_mark", 4: "D_labrax_no_mark", 10: "D_labrax_red", 11: "D_labrax_red", 13: "D_labrax_no_mark", 18: "D_labrax_red", 19: "D_labrax_no_mark", 20: "None", 31: "D_labrax_no_mark", 32: "D_labrax_red"},
        "10-07-59_0-r50-end": {-1: "None", 1: "D_labrax_no_mark", 2: "D_labrax_red", 3: "D_labrax_no_mark", 5: "D_labrax_no_mark", 6: "D_labrax_red", 9: "D_labrax_no_mark", 12: "D_labrax_no_mark", 13: "None", 14: "None", 15: "D_labrax_red", 16: "D_labrax_red", 18: "D_labrax_red", 19: "D_labrax_red", 20: "D_labrax_red", 21: "D_labrax_no_mark", 22: "None", 24: "D_labrax_red", 27: "D_labrax_no_mark", 28: "D_labrax_red"},
        "10-09-31_0-r0-344": {1: "D_labrax_no_mark", 4: "D_labrax_red", 5: "D_labrax_no_mark", 7: "D_labrax_no_mark", 8: "None", 9: "D_labrax_red", 10: "D_labrax_no_mark", 12: "D_labrax_no_mark", 16: "D_labrax_red", 17: "D_labrax_no_mark"},
        "10-09-31_0-r398-end": {-1: "None", 1: "D_labrax_red", 2: "D_labrax_no_mark", 4: "D_labrax_no_mark", 5: "D_labrax_no_mark", 7: "D_labrax_red", 8: "D_labrax_no_mark", 9: "D_labrax_no_mark", 12: "D_labrax_red", 13: "D_labrax_red", 14: "D_labrax_red"},
        "10-11-02_0-r0-244": {-1: "None", 1: "D_labrax_no_mark", 2: "D_labrax_red", 3: "D_labrax_red", 7: "D_labrax_no_mark", 9: "None", 14: "D_labrax_red"},
        "10-11-02_0-r295-end": {1: "D_labrax_red", 2: "D_labrax_no_mark", 4: "D_labrax_red", 8: "None", 10: "D_labrax_red", 14: "D_labrax_no_mark"},
        "10-12-34_0-r215-end": {1: "D_labrax_red", 2: "D_labrax_red", 4: "D_labrax_no_mark", 6: "D_labrax_red", 7: "None", 10: "D_labrax_red", 12: "D_labrax_red", 14: "None", 15: "D_labrax_no_mark"},
        "10-14-05_0-r0-528": {-1: "None", 1: "D_labrax_red", 2: "D_labrax_no_mark", 6: "D_labrax_red", 9: "D_labrax_no_mark", 10: "D_labrax_red", 12: "None", 13: "D_labrax_no_mark", 15: "None", 16: "None", 17: "D_labrax_no_mark", 18: "D_labrax_red"},
        "10-15-37_0-r496-end": {1: "D_labrax_no_mark", 2: "D_labrax_red", 5: "D_labrax_red", 6: "D_labrax_no_mark"},
        "10-18-24_0_compressed-r47-end": {-1: "None", 1: "D_labrax_red", 5: "D_labrax_no_mark", 7: "D_labrax_no_mark", 10: "D_labrax_red", 11: "None", 12: "D_labrax_red", 17: "D_labrax_red", 18: "D_labrax_red"},
        "10-19-44_0_compressed-r383-534": {1: "D_labrax_no_mark", 2: "D_labrax_red"},
        "10-19-44_0_compressed-r588-end": {1: "D_labrax_red", 2: "D_labrax_no_mark"},
        "10-20-29_1_compressed-r0-78": {-1: "None", 1: "D_labrax_red", 2: "D_labrax_no_mark", 3: "D_labrax_no_mark", 4: "D_labrax_red"},
        "10-22-00_1_compressed": {-1: "None", 1: "D_labrax_red", 2: "D_labrax_no_mark", 3: "D_labrax_red", 5: "None"},
        "10-37-15_1-r": {-1: "None", 1: "D_labrax_no_mark", 2: "D_labrax_red", 3: "D_labrax_red", 6: "D_labrax_red", 8: "D_labrax_no_mark", 9: "error", 10: "error"},
        "13-05-55_0-r": {1: "D_labrax_red", 2: "D_labrax_no_mark", 3: "D_labrax_black", 4: "D_labrax_black", 5: "D_labrax_no_mark", 8: "D_labrax_black", 10: "D_labrax_red", 11: "None", 15: "D_labrax_black", 20: "D_labrax_red", 22: "None", 25: "None", 26: "D_labrax_no_mark", 27: "None", 28: "D_labrax_no_mark", 30: "D_labrax_no_mark", 34: "D_labrax_no_mark", 38: "D_labrax_red", 39: "None", 42: "D_labrax_red", 47: "D_labrax_no_mark", 49: "D_labrax_black"},
        "13-22-22_0_compressed-r60-297": {1: "D_labrax_no_mark", 2: "D_labrax_black", 3: "D_labrax_red", 6: "D_labrax_red", 7: "None", 9: "D_labrax_red", 11: "None", 13: "D_labrax_black"},
        "13-22-22_0_compressed-r378-463": {1: "D_labrax_black", 2: "D_labrax_no_mark", 3: "D_labrax_red", 4: "error"},
        "13-23-54_0_compressed-r0-304": {-1: "None", 2: "D_labrax_red", 3: "D_labrax_black", 5: "D_labrax_no_mark"},
        "13-23-54_0_compressed-r450-524": {1: "D_labrax_no_mark", 2: "D_labrax_black", 3: "D_labrax_red", 5: "None"},
        "13-23-54_0_compressed-r602-end": {1: "D_labrax_red", 2: "D_labrax_black", 3: "D_labrax_no_mark"}
    }
}


SINGLE_FISH_GT = {
    "2024_11_12": "D_labrax_0", 
    "2024_11_28": "S_scombrus", 
    "2025_05_08": "D_labrax_1", 
    "2025_08_21": "D_labrax_red" # <--- ARREGLADO EL TYPO AQUÍ
}

FISH_MEASUREMENTS_CM = {
    # 2024_11_12
    "D_labrax_0": 31.5,
    
    # 2024-11-28
    "fish_1": 19.4,
    "caballa": 28.9,      
    "fish_3": 21.6,       
    
    # 2025-05-08
    "llobarro": 33.5,     
    "ochoa_cinta": 33.5,  
    "ochoa_no_cinta": 30.0, 
    "ochoa_petita": 30.0,   
    
    # 2025-08-21
    "D_labrax_red": 29.1,
    "D_labrax_black": 26.6,
    "D_labrax_no_mark": 32.3
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def extract_fishes_from_gt(bag_path):
    """
    Dynamically reconstructs the GT key from the bag file name, 
    then checks both Single and Multiple Fish dictionaries to find matches.
    """
    bag_name = bag_path.name
    bag_str = str(bag_path)
    
    # 1. Extract date: YYYY_MM_DD
    date_match = re.search(r"202\d[_-]\d{2}[_-]\d{2}", bag_str)
    if not date_match:
        return "Unknown Date", "Unknown"
    
    date_key = date_match.group(0).replace("-", "_")
    
    # 2. Reconstruct GT Key (e.g. 13-23-54_0_compressed-r450-524)
    name_no_ext = bag_name.replace(".bag", "")
    gt_key = None
    
    # Regex to pull apart the prefix (e.g., r0-244) and the time part (e.g., 10-11-02_0_compressed)
    match = re.search(r"(.*?)_?stereo_camera_images_\d{4}-\d{2}-\d{2}-(.+)", name_no_ext)
    
    if match:
        prefix = match.group(1)       # e.g. "r450-524" or "r" or ""
        time_part = match.group(2)    # e.g. "13-23-54_0_compressed"
        
        # If there's a prefix, append it to the end to match your dictionary format
        if prefix:
            gt_key = f"{time_part}-{prefix}"
        else:
            gt_key = time_part
    else:
        # Fallback for old file names without 'stereo_camera_images' string
        time_match = re.search(r"\d{2}-\d{2}-\d{2}_\d(_compressed)?", name_no_ext)
        if time_match:
            gt_key = time_match.group(0)

    fishes = set()
    detected_type = "Unknown"
    
    # 3. Check if it's in MULTIPLE_FISH_GT
    if date_key in MULTIPLE_FISH_GT and gt_key in MULTIPLE_FISH_GT[date_key]:
        detected_type = "multiple_fish"
        for fish_name in MULTIPLE_FISH_GT[date_key][gt_key].values():
            if str(fish_name).lower() not in ["error", "none"]:
                length = FISH_MEASUREMENTS_CM.get(fish_name, "N/A")
                fishes.add(f"{fish_name} ({length}cm)")
                
    # 4. Check if it's in SINGLE_FISH_GT
    elif date_key in SINGLE_FISH_GT:
        detected_type = "single_fish"
        fish_name = SINGLE_FISH_GT[date_key]
        length = FISH_MEASUREMENTS_CM.get(fish_name, "N/A")
        fishes.add(f"{fish_name} ({length}cm)")
            
    if fishes:
        return ", ".join(sorted(list(fishes))), detected_type
    else:
        # If it fails, print the constructed key so you can see exactly why it didn't match
        return f"Not found in GT (Searched for key: '{gt_key}')", detected_type

# ==========================================
# MAIN FUNCTION
# ==========================================
def generate_dataset_summary(root_folder, left_base_topic, right_base_topic, dataset_id):
    """
    Scans for .bag files, dynamically detects raw vs compressed topics,
    determines dataset type from path, calculates metrics, and generates summary.
    """
    root_dir = Path(root_folder)
    print("ROOT DIR: ", root_dir)
    bag_files = list(root_dir.rglob("*.bag"))
    
    if not bag_files:
        print(f"❌ No .bag files found in directory: {root_dir}")
        return
        
    print(f"🔍 Analyzing {len(bag_files)} bagfiles... This might take a while.\n")
    
    data = []
    
    for bag_path in bag_files:
        try:
            # 2. Get File Size in GB
            size_gb = bag_path.stat().st_size / (1024 ** 3)
            
            # 3. Extract Date and Time from path/filename
            # Extract Date: 2025-08-21
            date_match = re.search(r"202\d[_-]\d{2}[_-]\d{2}", bag_path.name)
            bag_date = date_match.group(0).replace("-", "_") if date_match else "Unknown"
            
            # Extract Time: Looks for HH-MM-SS right before the underscore (e.g., 13-35-09_1)
            time_match = re.search(r"(\d{2}-\d{2}-\d{2})_\d", bag_path.name)
            bag_time = time_match.group(1) if time_match else "Unknown"

            # 4. Read ROS Bag metadata (Fast index check)
            with rosbag.Bag(bag_path, 'r') as bag:
                start_time = bag.get_start_time()
                end_time = bag.get_end_time()
                duration_sec = end_time - start_time
                
                # Check available topics in this specific bag
                topics_info = bag.get_type_and_topic_info()[1]
                
                left_raw = left_base_topic
                left_comp = left_base_topic + "/compressed"
                right_raw = right_base_topic
                right_comp = right_base_topic + "/compressed"
                
                num_left = 0
                num_right = 0
                image_format = "Unknown"

                # Check Left Camera Format
                if left_raw in topics_info:
                    num_left = topics_info[left_raw].message_count
                    image_format = "Raw"
                elif left_comp in topics_info:
                    num_left = topics_info[left_comp].message_count
                    image_format = "Compressed"

                # Check Right Camera
                if right_raw in topics_info:
                    num_right = topics_info[right_raw].message_count
                elif right_comp in topics_info:
                    num_right = topics_info[right_comp].message_count
                
                # Calculate FPS (using left topic as reference)
                fps = (num_left / duration_sec) if duration_sec > 0 else 0.0
                
                # 5. Get Fishes from Ground Truth 
                # (Recibe las dos variables que devuelve tu genial función)
                fishes_present, dataset_type = extract_fishes_from_gt(bag_path)
                
                data.append({
                    "Dataset_ID": dataset_id,
                    "Bagfile_Name": bag_path.name,
                    "Date": bag_date,
                    "Time": bag_time,
                    "Dataset_Type": dataset_type,
                    "Image_Format": image_format,
                    "Duration_Sec": round(duration_sec, 2),
                    "Num_Images_Left": num_left,
                    "Num_Images_Right": num_right,
                    "Frame_Rate_FPS": round(fps, 2),
                    "Size_GB": round(size_gb, 2),
                    "Fishes_Present": fishes_present
                })
                
                print(f"   ✅ {bag_path.name} processed. ({dataset_type} | {image_format})")
                
        except Exception as e:
            print(f"   ❌ Error reading {bag_path.name}: {e}")

    # ==========================================
    # 📊 CREATE DATAFRAME & GLOBAL STATISTICS
    # ==========================================
    df_summary = pd.DataFrame(data)
    
    if df_summary.empty:
        print("❌ Could not extract information from any bagfile.")
        return

    # Global Calculations
    total_bags = len(df_summary)
    total_seconds = df_summary["Duration_Sec"].sum()
    total_gb = df_summary["Size_GB"].sum()
    
    total_hours = total_seconds / 3600
    total_minutes = total_seconds / 60
    
    print("\n" + "="*95)
    print("📈 DETAILED BAGFILE SUMMARY")
    print("="*95)
    display_cols = ["Dataset_ID", "Bagfile_Name", "Time", "Dataset_Type", "Duration_Sec", "Fishes_Present"]
    print(df_summary[display_cols].to_string(index=False))
    
    print("\n" + "="*95)
    print("🌍 GLOBAL DATASET QUANTIFICATION")
    print("="*95)
    print(f"   ▶ Dataset ID:          {dataset_id}")
    print(f"   ▶ Total Bagfiles:      {total_bags}")
    print(f"   ▶ Total Duration:      {total_hours:.2f} hours ({total_minutes:.2f} minutes)")
    print(f"   ▶ Total Size:          {total_gb:.2f} GB")
    print("="*95 + "\n")
    
    # Save to CSV
    csv_out = root_dir / f"Dataset_Summary_{dataset_id.replace(' ', '_')}.csv"
    df_summary.to_csv(csv_out, index=False, sep=",")
    print(f"💾 Summary successfully saved to: {csv_out}")


# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    # Update this to point exactly where your 2025_08_21 bagfiles are
    ROOT_FOLDER = "//media/slimbook/2FE6-5ED9/bagfiles/LIMIA/retalls_compresssed/" 
    
    LEFT_BASE_TOPIC = "/stereo_ch3/left/image_raw" 
    RIGHT_BASE_TOPIC = "/stereo_ch3/right/image_raw"
    DATASET_ID = "Dataset_B"
    
    generate_dataset_summary(ROOT_FOLDER, LEFT_BASE_TOPIC, RIGHT_BASE_TOPIC, DATASET_ID)