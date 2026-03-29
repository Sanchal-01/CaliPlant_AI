import os
import time
from icrawler.builtin import BingImageCrawler


IMAGE_LIMIT = 50  # This is the number of images we would have per plant.
BASE_DATA_DIR = "03_Data/train"

def smart_plant_downloader(plant_name):   
    safe_name = plant_name.replace(' ', '_')  # Folder name should match exactly with what we created.
    folder_path = os.path.join(BASE_DATA_DIR, safe_name)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f" Naya folder banaya: {safe_name}")

    existing_files = len(os.listdir(folder_path))
    if existing_files >= IMAGE_LIMIT:
        print(f" {plant_name} already downloaded ({existing_files} images), Skipping.....")
        return

    print(f" Finding '{plant_name}' photos...")
    
    try:
        crawler = BingImageCrawler(
            storage={'root_dir': folder_path},
            downloader_threads=4 # Thoda fast download ke liye threads badha diye hain. Threads allow us to simultaneously download photos here. This increases overall speed.
        )
        
        # Search query ko thoda specific banaya hai taaki noise kam kar sake.
        search_query = f"{plant_name} plant California"
        crawler.crawl(keyword=search_query, max_num=IMAGE_LIMIT)
        
        print(f" {plant_name} ka batch successfully khatam!")
        time.sleep(2) #  2 sec ka delay kaafi hai.
        
    except Exception as e:
        print(f"some issue {plant_name}: {e}")


if __name__ == "__main__":
    # Nayi list
    new_plants = [
        "Poison Hemlock", "Matilija Poppy", "Stinknet", "Sticky Monkey-flower", 
        "Prickly Pear", "Mustard Plant", "Manzanita", "Toyon", "Chalk Dudleya", 
        "Ice Plant", "Blue Gum Eucalyptus", "Tree Tobacco", "Our Lords Candle", 
        "Common Fennel", "California Buckwheat", "Black Sage", "Ceanothus CA Lilac", 
        "Mariposa Lily", "Western Sycamore", "Palo Verde", "Castor Bean", 
        "Wild Radish", "Blue Elderberry", "Bladderpod", "CA Bush Sunflower", 
        "Meyer Lemon", "Bougainvillea", "Jacaranda", "Bird of Paradise", 
        "Rosemary", "European Olive", "French Lavender", "Snake Plant", 
        "Spider Plant", "Jade Plant", "Aloe Vera", "Swiss Cheese Plant", 
        "ZZ Plant", "Golden Pothos", "Fiddle Leaf Fig", "Indoor Rose", 
        "Peace Lily", "Rubber Plant", "Flamingo Flower", "Heartleaf Philodendron", "English Ivy"
    ]

    print(f"--- CaliPlant AI: {len(new_plants)} Naye Plants ka Data Collection ---")
    
    for plant in new_plants:
        smart_plant_downloader(plant)
        
    print("\n Sab done!!!! Saare 46 naye folders ab images se bhar chuke hain.")