import os
import random
import shutil

# This is Path to my training & validation data:
TRAIN_DIR = "03_Data/train"
VAL_DIR = "03_Data/validation"
SPLIT_RATIO = 0.2 

def split_data():
    # Identify all plant categories (folders)
    plants = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    
    print(f" Found {len(plants)} plant classes. Starting split...")

    for plant in plants:
        train_plant_path = os.path.join(TRAIN_DIR, plant)
        val_plant_path = os.path.join(VAL_DIR, plant)
        
        # Create corresponding validation folder
        if not os.path.exists(val_plant_path):
            os.makedirs(val_plant_path)
        
        images = [f for f in os.listdir(train_plant_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(images) > 0:
            num_to_move = int(len(images) * SPLIT_RATIO)
            # Ensure at least 1 image is moved if the folder isn't empty
            num_to_move = max(1, num_to_move) if len(images) >= 5 else 0
            
            images_to_move = random.sample(images, num_to_move)
            
            for img in images_to_move:
                shutil.move(os.path.join(train_plant_path, img), os.path.join(val_plant_path, img))
    
    print(f" Splitting complete.")

if __name__ == "__main__":
    split_data()