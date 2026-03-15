import os
import time
from icrawler.builtin import BingImageCrawler

# --- SETTINGS ---
# Kitni images chahiye har plant ki
IMAGE_LIMIT = 15 
# Folder jahan data save hoga
BASE_DATA_DIR = "03_Data/train"

def smart_plant_downloader(plant_name):
    # 1. Folder ka naam set karo (Space ki jagah Underscore _)
    safe_name = plant_name.replace(' ', '_')
    folder_path = os.path.join(BASE_DATA_DIR, safe_name)
    
    # 2. Agar folder nahi hai toh banao
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"📁 Naya folder banaya: {safe_name}")

    # 3. SMART CHECK: Agar images pehle se hain toh skip karo
    # Isse tera time aur internet dono bachega
    existing_files = len(os.listdir(folder_path))
    if existing_files >= IMAGE_LIMIT:
        print(f"⏩ {plant_name} pehle se downloaded hai ({existing_files} images). Skipping...")
        return

    print(f"🚀 Bing se '{plant_name}' ki photos dhoond raha hoon...")
    
    try:
        # 4. Bing Crawler Setup
        # 'icrawler' auto-handle karta hai headers taaki rate limit na aaye
        crawler = BingImageCrawler(
            storage={'root_dir': folder_path},
            downloader_threads=2 # Zyada threads matlab thoda fast, par safe
        )
        
        # 5. Actual Download Command
        crawler.crawl(keyword=plant_name, max_num=IMAGE_LIMIT)
        
        print(f"✅ {plant_name} ka batch successfully khatam!")
        
        # 6. HUMAN-LIKE PAUSE: 
        # Ek plant ke baad 5 second ka break taaki server ko robot na lage
        time.sleep(5) 
        
    except Exception as e:
        print(f"⚠️ Bhai kuch gadbad hui {plant_name} mein: {e}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Didi ke project ke liye final list
    cali_plants = [
        "California Poppy", 
        "Poison Oak", 
        "Joshua Tree", 
        "Coast Live Oak", 
        "California Sagebrush"
    ]

    print("--- 🌿 CaliPlant AI Scanner: Data Collection Shuru ---")
    
    for plant in cali_plants:
        smart_plant_downloader(plant)
        
    print("\n🎉 Sab done! Ab tera '03_Data/train' folder AI training ke liye ready hai.")