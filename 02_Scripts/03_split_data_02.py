import os
import random
import shutil

# Root path configuration
TRAIN_DIR = "03_Data/train"
VAL_DIR = "03_Data/validate"
TEST_DIR = "03_Data/test"      # Experience from KrishiNetra: Added test directory path here inorder to check model performance on untouched data set images.

# Ratios relative to the original entire dataset
VAL_RATIO = 0.20    # 20%
TEST_RATIO = 0.10   # 10%

def split_data():
    # Identify all plant categories (folders) based on what's currently in training
    plants = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    
    print(f" Found {len(plants)} plant classes. Starting 70-20-10 split...now")

    for plant in plants:
        train_plant_path = os.path.join(TRAIN_DIR, plant)
        val_plant_path = os.path.join(VAL_DIR, plant)
        test_plant_path = os.path.join(TEST_DIR, plant)
        
        # 1. Create target validation and test directories if they don't exist
        if not os.path.exists(val_plant_path):
            os.makedirs(val_plant_path)
        if not os.path.exists(test_plant_path):
            os.makedirs(test_plant_path)
        
        # Grab current images inside the train folder
        images = [f for f in os.listdir(train_plant_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        total_images = len(images)
        
        if total_images >= 10:  # Ensures there are enough images to divide meaningfully
            # Calculate absolute numbers based on original total size
            num_test = int(total_images * TEST_RATIO)
            num_val = int(total_images * VAL_RATIO)
            
            # STEP 1: Move images to TEST 
            test_images = random.sample(images, num_test)
            for img in test_images:
                shutil.move(os.path.join(train_plant_path, img), os.path.join(test_plant_path, img))
                images.remove(img)  # Remove from available pool so we don't pick it again
                
            # STEP 2: Move images to VALIDATION
            val_images = random.sample(images, num_val)
            for img in val_images:
                shutil.move(os.path.join(train_plant_path, img), os.path.join(val_plant_path, img))
            
            print(f"  [{plant}]: Total: {total_images} -> Train: {total_images - num_test - num_val}, Val: {num_val}, Test: {num_test}")
        else:
            print(f" Skipping [{plant}] folder because it has too few images ({total_images}) to split reliably.")
            
    print(f"\n Splitting complete! Our data is now structured as 70/20/10.")

if __name__ == "__main__":
    split_data()