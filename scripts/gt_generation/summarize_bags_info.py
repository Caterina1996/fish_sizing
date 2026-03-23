import rosbag
import pandas as pd
from pathlib import Path

def generar_resum_dataset(carpeta_arrel, camera_topic="/stereo_ch3/left/camera_info"):
    """
    Analitza tots els .bag, calcula durades i FPS, i genera una taula i estadístiques globals.
    """
    root_dir = Path(carpeta_arrel)
    bag_files = list(root_dir.rglob("*.bag"))
    
    if not bag_files:
        print(f"❌ No s'han trobat arxius .bag a la carpeta: {root_dir}")
        return
        
    print(f"🔍 Analitzant {len(bag_files)} bagfiles (buscant el tòpic {camera_topic})... Això pot trigar una mica.\n")
    
    dades = []
    
    for bag_path in bag_files:
        try:
            with rosbag.Bag(bag_path, 'r') as bag:
                start_time = bag.get_start_time()
                end_time = bag.get_end_time()
                
                durada_segons = end_time - start_time
                
                # Comptar els frames (missatges) només d'aquest tòpic per treure els FPS reals
                num_frames = bag.get_message_count(camera_topic)
                
                # Evitar divisions per zero si el bag està corrupte o buit
                fps = (num_frames / durada_segons) if durada_segons > 0 else 0.0
                
                dades.append({
                    "Bagfile_ID": bag_path.name,
                    "Durada_segons": round(durada_segons, 2),
                    "Num_Frames": num_frames,
                    "Frame_Rate_FPS": round(fps, 2)
                })
                
                print(f"   ✅ {bag_path.name} llegit.")
                
        except Exception as e:
            print(f"   ❌ Error llegint {bag_path.name}: {e}")

    # ==========================================
    # 📊 CREACIÓ DE LA TAULA I ESTADÍSTIQUES
    # ==========================================
    df_resum = pd.DataFrame(dades)
    
    if df_resum.empty:
        print("❌ No s'ha pogut extreure informació de cap bagfile.")
        return

    # Càlculs globals
    total_bags = len(df_resum)
    total_segons = df_resum["Durada_segons"].sum()
    mitjana_durada = df_resum["Durada_segons"].mean()
    mitjana_fps = df_resum["Frame_Rate_FPS"].mean()
    
    hores_totals = total_segons / 3600
    minuts_totals = total_segons / 60
    
    print("\n" + "="*70)
    print("📈 TAULA DE DETALL PER BAGFILE")
    print("="*70)
    # Mostrem la taula per consola (es veurà molt bé a Jupyter o a la terminal)
    print(df_resum.to_string(index=False))
    
    print("\n" + "="*70)
    print("🌍 QUANTIFICACIÓ GLOBAL DEL DATASET")
    print("="*70)
    print(f"   ▶ Total Bagfiles:      {total_bags}")
    print(f"   ▶ Durada TOTAL:        {hores_totals:.2f} hores ({minuts_totals:.2f} minuts)")
    print(f"   ▶ Mitjana de durada:   {mitjana_durada:.2f} segons per vídeo")
    print(f"   ▶ Mitjana Frame Rate:  {mitjana_fps:.2f} FPS")
    print("="*70 + "\n")
    
    # Guardar a CSV de forma automàtica
    csv_out = root_dir / "Resum_Dataset_Bagfiles.csv"
    df_resum.to_csv(csv_out, index=False)
    print(f"💾 Taula guardada correctament a: {csv_out}")

# ==========================================
# EXEMPLE D'ÚS
# ==========================================
if __name__ == "__main__":
    # Canvia-ho per la teva ruta
    CARPETA_BAGS = "//media/slimbook/easystore1/bagfiles/seleccio_article/2025_08_21/lanty_1/1_peix" 
    
    # ⚠️ IMPORTANT: Assegura't de posar el nom exacte del tòpic de la càmera (ex: "/camera/color/image_raw" o "/left/image_raw")
    TOPIC_CAMERA = "/stereo_ch3/left/camera_info" 
    
    generar_resum_dataset(CARPETA_BAGS, TOPIC_CAMERA)