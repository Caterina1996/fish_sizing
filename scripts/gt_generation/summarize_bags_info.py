import rosbag
import pandas as pd
from pathlib import Path
import re

# ==========================================
# GROUND TRUTH DICTIONARY
# ==========================================
MULTIPLE_FISH_GT = {
    "2024_11_28": {
        '13-14-43_0': {1: "caballa", 21: "fish_3", 24: "caballa", 29: "fish_3", 37: "caballa", 45: "caballa", 58: "fish_3", 62: "fish_3", 75: "fish_3", 77: "fish_3", 81: "fish_3", 82: "fish_3"},
        '13-16-54_0': {1: "caballa", 12: "fish_3", 17: "fish_3", 29: "fish_3", 41: "caballa", 49: "caballa", 53: "caballa", 59: "fish_3", 67: "fish_3", 68: "fish_3", 70: "caballa", 71: "fish_3"},
        '13-27-52_0': {1: "caballa", 2: "fish_3", 3: "fish_3", 5: "fish_3", 6: "fish_3", 8: "fish_3", 11: "caballa", 12: "fish_3", 13: "fish_3", 33: "caballa", 35: "error", 37: "caballa"},
        '13-32-16_0': {1: "fish_3", 4: "caballa", 6: "caballa", 17: "fish_3", 25: "caballa", 28: "caballa", 30: "caballa", 44: "caballa", 52: "caballa", 53: "caballa", 67: "caballa"},
        '13-34-27_0': {1: "caballa", 5: "caballa", 8: "caballa", 16: "caballa", 21: "caballa", 23: "fish_3", 28: "caballa", 30: "error", 31: "caballa"},
        '13-36-49_0': {1: "fish_3", 2: "caballa", 3: "fish_3", 6: "fish_3", 10: "fish_3", 11: "fish_3", 19: "fish_3", 20: "caballa", 24: "fish_3", 30: "caballa", 33: "fish_3", 40: "error", 41: "error", 42: "error"},
        '13-39-00_0': {1: "error", 2: "caballa", 8: "caballa", 11: "caballa", 12: "caballa", 23: "error"},
        '13-41-12_0': {2: "fish_3", 3: "caballa", 11: "caballa"},
        '13-42-45_0': {1: "caballa", 2: "fish_3", 5: "caballa", 17: "caballa", 26: "caballa"},
        '13-48-24_0': {1: "caballa", 4: "fish_3", 5: "fish_3", 10: "fish_3"},
        '13-49-20_0': {1: "caballa", 2: "fish_3", 9: "fish_3", 13: "caballa", 16: "fish_3", 25: "caballa", 29: "fish_3", 32: "fish_3", 34: "fish_3", 36: "fish_3", 41: "fish_3", 44: "fish_1", 56: "fish_1", 58: "fish_3", 60: "fish_1", 61: "fish_3", 64: "fish_3", 69: "fish_1", 72: "fish_1", 73: "caballa", 75: "fish_1", 76: "fish_1", 77: "fish_1"}
    },
    "2025_05_08": {
        '11-28-42_0': {1: "ochoa_petita", 2: "llobarro", 5: "ochoa_petita"},
        '11-29-16_1': {1: "llobarro", 2: "ochoa_petita", 6: "ochoa_petita", 8: "ochoa_petita"},
        '11-29-49_2': {1: "ochoa_petita", 2: "llobarro", 6: "ochoa_petita", 7: "ochoa_petita"},
        '11-30-34_0': {1: "ochoa_petita", 6: "llobarro"},
        '11-31-08_1': {1: "llobarro", 2: "ochoa_petita", 3: "ochoa_petita", 8: "ochoa_petita"},
        '11-33-23_0': {1: "llobarro", 2: "ochoa_petita", 10: "ochoa_petita", 18: "llobarro", 19: "ochoa_petita", 20: "ochoa_petita"},
        '11-33-52_1': {1: "llobarro", 2: "ochoa_petita", 4: "llobarro"},
        '11-36-36_0': {1: "llobarro", 2: "ochoa_petita", 8: "llobarro", 13: "ochoa_petita", 23: "ochoa_petita", 27: "ochoa_petita", 30: "llobarro"}, 
        '11-37-20_1': {1: "error", 2: "llobarro", 8: "ochoa_petita", 16: "llobarro", 18: "ochoa_petita", 25: "ochoa_petita", 33: "ochoa_petita", 34: "ochoa_petita"},
        '11-41-45_0': {1: "llobarro", 8: "llobarro"},
        '11-42-15_1': {1: "llobarro"},
        '11-44-48_0': {1: "llobarro", 3: "ochoa_petita", 6: "ochoa_petita", 7: "ochoa_petita", 10: "error"},
        '11-48-15_0': {1: "llobarro", 2: "ochoa_no_cinta", 4: "ochoa_cinta", 6: "ochoa_cinta", 13: "ochoa_cinta", 14: "llobarro", 16: "ochoa_no_cinta", 18: "ochoa_cinta", 19: "ochoa_cinta", 22: "ochoa_cinta"},
        '11-49-52_0': {1: "llobarro", 3: "ochoa_no_cinta", 4: "ochoa_cinta", 7: "ochoa_cinta", 10: "llobarro", 13: "ochoa_no_cinta", 14: "ochoa_cinta"},
        '11-57-17_0': {1: "llobarro", 2: "ochoa_cinta", 4: "ochoa_no_cinta"},
        '11-57-59_0': {1: "ochoa_no_cinta", 2: "ochoa_no_cinta", 4: "ochoa_cinta", 9: "llobarro", 10: "llobarro", 16: "ochoa_no_cinta", 20: "ochoa_cinta", 24: "llobarro"},
        '11-59-57_0': {1: "ochoa_no_cinta", 2: "ochoa_cinta", 8: "llobarro", 14: "llobarro", 20: "ochoa_no_cinta", 24: "llobarro"},
        '12-00-39_0': {1: "ochoa_cinta", 2: "llobarro", 3: "ochoa_no_cinta", 14: "ochoa_cinta", 17: "ochoa_no_cinta", 29: "ochoa_no_cinta", 34: "ochoa_no_cinta", 36: "ochoa_no_cinta", 39: "ochoa_no_cinta"},
        '12-01-20_0': {1: "ochoa_no_cinta", 2: "ochoa_cinta", 3: "llobarro", 4: "ochoa_no_cinta", 5: "llobarro", 9: "ochoa_no_cinta", 13: "ochoa_cinta", 26: "llobarro", 29: "ochoa_cinta", 33: "ochoa_cinta"},
        '12-02-02_0': {1: "ochoa_cinta", 2: "ochoa_no_cinta", 3: "llobarro", 9: "ochoa_no_cinta", 11: "ochoa_cinta", 13: "ochoa_cinta", 16: "ochoa_no_cinta"},
        '12-02-44_0': {1: "llobarro", 2: "ochoa_no_cinta", 3: "ochoa_cinta", 11: "ochoa_cinta", 13: "ochoa_no_cinta"}
    },
    "2025_08_21": {
        "10-01-53_0_compressed-r201-end": {-1: "None", 1: "red", 3: "no_mark", 4: "no_mark", 10: "red", 11: "red", 13: "no_mark", 18: "red", 19: "no_mark", 20: "None", 31: "no_mark", 32: "red"},
        "10-07-59_0-r50-end": {-1: "None", 1: "no_mark", 2: "red", 3: "no_mark", 5: "no_mark", 6: "red", 9: "no_mark", 12: "no_mark", 13: "None", 14: "None", 15: "red", 16: "red", 18: "red", 19: "red", 20: "red", 21: "no_mark", 22: "None", 24: "red", 27: "no_mark", 28: "red"},
        "10-09-31_0-r0-344": {1: "no_mark", 4: "red", 5: "no_mark", 7: "no_mark", 8: "None", 9: "red", 10: "no_mark", 12: "no_mark", 16: "red", 17: "no_mark"},
        "10-09-31_0-r398-end": {-1: "None", 1: "red", 2: "no_mark", 4: "no_mark", 5: "no_mark", 7: "red", 8: "no_mark", 9: "no_mark", 12: "red", 13: "red", 14: "red"},
        "10-11-02_0-r0-244": {-1: "None", 1: "no_mark", 2: "red", 3: "red", 7: "no_mark", 9: "None", 14: "red"},
        "10-11-02_0-r295-end": {1: "red", 2: "no_mark", 4: "red", 8: "None", 10: "red", 14: "no_mark"},
        "10-12-34_0-r215-end": {1: "red", 2: "red", 4: "no_mark", 6: "red", 7: "None", 10: "red", 12: "red", 14: "None", 15: "no_mark"},
        "10-14-05_0-r0-528": {-1: "None", 1: "red", 2: "no_mark", 6: "red", 9: "no_mark", 10: "red", 12: "None", 13: "no_mark", 15: "None", 16: "None", 17: "no_mark", 18: "red"},
        "10-15-37_0-r496-end": {1: "no_mark", 2: "red", 5: "red", 6: "no_mark"},
        "10-18-24_0_compressed-r47-end": {-1: "None", 1: "red", 5: "no_mark", 7: "no_mark", 10: "red", 11: "None", 12: "red", 17: "red", 18: "red"},
        "10-19-44_0_compressed-r383-534": {1: "no_mark", 2: "red"},
        "10-19-44_0_compressed-r588-end": {1: "red", 2: "no_mark"},
        "10-20-29_1_compressed-r0-78": {-1: "None", 1: "red", 2: "no_mark", 3: "no_mark", 4: "red"},
        "10-22-00_1_compressed": {-1: "None", 1: "red", 2: "no_mark", 3: "red", 5: "None"},
        "10-37-15_1-r": {-1: "None", 1: "no_mark", 2: "red", 3: "red", 6: "red", 8: "no_mark", 9: "error", 10: "error"},
        "13-05-55_0-r": {1: "red", 2: "no_mark", 3: "black", 4: "black", 5: "no_mark", 8: "black", 10: "red", 11: "None", 15: "black", 20: "red", 22: "None", 25: "None", 26: "no_mark", 27: "None", 28: "no_mark", 30: "no_mark", 34: "no_mark", 38: "red", 39: "None", 42: "red", 47: "no_mark", 49: "black"},
        "13-22-22_0_compressed-r60-297": {1: "no_mark", 2: "black", 3: "red", 6: "red", 7: "None", 9: "red", 11: "None", 13: "black"},
        "13-22-22_0_compressed-r378-463": {1: "black", 2: "no_mark", 3: "red", 4: "error"},
        "13-23-54_0_compressed-r0-304": {-1: "None", 2: "red", 3: "black", 5: "no_mark"},
        "13-23-54_0_compressed-r450-524": {1: "no_mark", 2: "black", 3: "red", 5: "None"},
        "13-23-54_0_compressed-r602-end": {1: "red", 2: "black", 3: "no_mark"}
    }
}


SINGLE_FISH_GT = {
    "2024_11_12": "D_labrax_0", "2024_11_28": "S_scombrus", 
    "2025_05_08": "D_labrax_1", "2025_08_21": "D_labrax_red"
}

FISH_MEASUREMENTS_CM = {
    # 2024_11_12
    "D_labrax_0": 31.5,
    
    # 2024-11-28
    "fish_1": 19.4,
    "caballa": 28.9,      # Corresponde a Fish 2 (Mackerel)
    "fish_3": 21.6,       # Corresponde a Fish 3 (Bogue)
    
    # 2025-05-08
    "llobarro": 33.5,     # Corresponde a Fish 4 (Seabass)
    "ochoa_cinta": 33.5,  # Corresponde a Fish 5 (Blue Whiting 1)
    "ochoa_no_cinta": 30.0, # Corresponde a Fish 6 (Blue Whiting 2)
    "ochoa_petita": 30.0,   # Alias de Fish 6
    
    # 2025-08-21
    "red": 29.1,
    "black": 26.6,
    "no_mark": 32.3
}


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def extract_fishes_from_gt(bag_path,dataset_type):
    """
    Attempts to extract the list of fishes from the GT dictionary
    based on the date and time embedded in the bag path.
    """
    bag_str = str(bag_path)
    
    # Extract date: YYYY_MM_DD or YYYY-MM-DD
    date_match = re.search(r"202\d[_-]\d{2}[_-]\d{2}", bag_str)
    if not date_match:
        return "Unknown Date"
    
    date_key = date_match.group(0).replace("-", "_")
    
    # Extract time identifier (e.g., 13-22-22_0)
    time_match = re.search(r"\d{2}-\d{2}-\d{2}_\d", bag_path.name)
    if not time_match:
        return "Unknown Time ID"
        
    time_key = time_match.group(0)
    
    fishes = set()
    
    if dataset_type == "multiple_fish":
        if date_key in MULTIPLE_FISH_GT:
            for gt_key, fish_dict in MULTIPLE_FISH_GT[date_key].items():
                if time_key in gt_key:
                    for fish_name in fish_dict.values():
                        # Ignoramos los "error" o "None" (pasando a minúsculas por seguridad)
                        if str(fish_name).lower() not in ["error", "none"]:
                            # Buscamos la medida exacta usando el nombre original
                            length = FISH_MEASUREMENTS_CM.get(fish_name, "N/A")
                            fishes.add(f"{fish_name} ({length}cm)")
                            
    elif dataset_type == "single_fish":
        if date_key in SINGLE_FISH_GT:
            fish_name = SINGLE_FISH_GT[date_key]
            # Buscamos la medida exacta
            length = FISH_MEASUREMENTS_CM.get(fish_name, "N/A")
            fishes.add(f"{fish_name} ({length}cm)")
            
    if fishes:
        return ", ".join(sorted(list(fishes)))
    else:
        return "Not found in GT"

# ==========================================
# MAIN FUNCTION
# ==========================================
def generate_dataset_summary(root_folder, left_base_topic, right_base_topic, dataset_type, dataset_id):
    """
    Scans for .bag files, dynamically detects raw vs compressed topics,
    calculates metrics, and generates a global summary Excel/CSV file.
    """
    root_dir = Path(root_folder)
    bag_files = list(root_dir.rglob("*.bag"))
    
    if not bag_files:
        print(f"❌ No .bag files found in directory: {root_dir}")
        return
        
    print(f"🔍 Analyzing {len(bag_files)} bagfiles... This might take a while.\n")
    
    data = []
    
    for bag_path in bag_files:
        try:
            # 1. Get File Size in GB
            size_gb = bag_path.stat().st_size / (1024 ** 3)
            
            # 2. Extract Date from path
            date_match = re.search(r"202\d[_-]\d{2}[_-]\d{2}", str(bag_path))
            bag_date = date_match.group(0).replace("-", "_") if date_match else "Unknown"

            # 3. Read ROS Bag metadata (Fast index check)
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
                
                # 4. Get Fishes from Ground Truth
                fishes_present = extract_fishes_from_gt(bag_path, dataset_type)
                
                # We removed Relative_Path and added Dataset_ID at the top
                data.append({
                    "Dataset_ID": dataset_id,
                    "Bagfile_Name": bag_path.name,
                    "Date": bag_date,
                    "Dataset_Type": dataset_type,
                    "Image_Format": image_format,
                    "Duration_Sec": round(duration_sec, 2),
                    "Num_Images_Left": num_left,
                    "Num_Images_Right": num_right,
                    "Frame_Rate_FPS": round(fps, 2),
                    "Size_GB": round(size_gb, 2),
                    "Fishes_Present": fishes_present
                })
                
                print(f"   ✅ {bag_path.name} processed. ({image_format})")
                
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
    # Updated display columns for the console printout
    display_cols = ["Dataset_ID", "Bagfile_Name", "Date", "Duration_Sec", "Size_GB", "Fishes_Present"]
    print(df_summary[display_cols].to_string(index=False))
    
    print("\n" + "="*95)
    print("🌍 GLOBAL DATASET QUANTIFICATION")
    print("="*95)
    print(f"   ▶ Dataset ID:          {dataset_id}")
    print(f"   ▶ Total Bagfiles:      {total_bags}")
    print(f"   ▶ Dataset Type:        {dataset_type}")
    print(f"   ▶ Total Duration:      {total_hours:.2f} hours ({total_minutes:.2f} minutes)")
    print(f"   ▶ Total Size:          {total_gb:.2f} GB")
    print("="*95 + "\n")
    
    # Save to CSV
    csv_out = root_dir / f"Dataset_Summary_{dataset_id.replace(' ', '_')}_{dataset_type}.csv"
    df_summary.to_csv(csv_out, index=False, sep=",")
    print(f"💾 Summary successfully saved to: {csv_out}")

# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    ROOT_FOLDER = "/media/slimbook/easystore1/bagfiles/seleccio_article/Piscina/2024_11_12/" 
    
    # Set ONLY the base topic (do NOT add '/compressed'). 
    # The script will automatically check for base and base+'/compressed'.
    LEFT_BASE_TOPIC = "/stereo_ch3/left/image_raw" 
    RIGHT_BASE_TOPIC = "/stereo_ch3/right/image_raw"
    
    # Set this to "single_fish" or "multiple_fish"
    DATASET_TYPE = "single_fish" 
    DATASET_ID = "Dataset 1"
    
    generate_dataset_summary(ROOT_FOLDER, LEFT_BASE_TOPIC, RIGHT_BASE_TOPIC, DATASET_TYPE, DATASET_ID)