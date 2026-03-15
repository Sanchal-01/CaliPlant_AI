import os
import requests
import time
from duckduckgo_search import DDGS

def download_plant_data(plant_name, limit=15):
    folder_path = f"03_Data/train/{plant_name.replace(' ', '_')}"
    
    # --- SMART CHECK LOGIC ---
    # Agar folder pehle se hai aur usme 10 se zyada photos hain, toh skip kar do
    if os.path.exists(folder_path) and len(os.listdir(folder_path)) >= limit:
        print(f"⏩ {plant_name} pehle se downloaded hai. Skipping...")
        return
    # --------------------------

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print(f"Searching for: {plant_name}...")
    
    try:
        with DDGS() as ddgs:
            results = ddgs.images(plant_name, max_results=limit)
            for i, res in enumerate(results):
                try:
                    img_url = res['image']
                    response = requests.get(img_url, timeout=10)
                    with open(f"{folder_path}/{i}.jpg", "wb") as handler:
                        handler.write(response.content)
                    print(f"  - Downloaded image {i+1}")
                    time.sleep(3) # Thoda extra delay safety ke liye
                except Exception:
                    continue 
    except Exception as e:
        print(f"⚠️ Error: {e}. Thodi der wait karein.")
                
    print(f"✅ {plant_name} ka kaam ho gaya!\n")

plant_list = ["California Poppy", "Joshua Tree", "Coast Live Oak", "Poison Oak"]

for plant in plant_list:
    download_plant_data(plant)
    time.sleep(5)