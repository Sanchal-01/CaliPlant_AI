import Augmentor
import os
from PIL import Image

# NOTE: Expects 'CALIPLANT_AI_Dataset' directory on Desktop containing raw species subfolders.
dataset_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'CALIPLANT_AI_Dataset')

# 2. Get a list of all 51 plant sub-folders automatically
if os.path.exists(dataset_path):
    folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    print(f" Nice! Found {len(folders)} plant species folders.\n")
else:
    print(f" Ahh.., couldn't find the folder at: {dataset_path}")
    folders = []

print(" Step 1: Cleaning up formats, fixing transparency & resizing to 224x224...")

# --- STEP 1: CLEANING & RESIZING ---
for folder in folders:
    full_folder_path = os.path.join(dataset_path, folder)
    
    for filename in os.listdir(full_folder_path):
        file_path = os.path.join(full_folder_path, filename)
        
        # Skip output subfolders if they exist, process only actual image files
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            try:
                with Image.open(file_path) as img:
                    # Resize every image to a clean 224x224
                    img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
                    
                    # Handles PNG/WebP alpha transparency so model training doesn't break
                    if img_resized.mode in ('RGBA', 'LA') or (img_resized.mode == 'P' and 'transparency' in img_resized.info):
                        bg = Image.new("RGB", img_resized.size, (255, 255, 255))
                        if img_resized.mode == 'RGBA':
                            bg.paste(img_resized, mask=img_resized.split()[3])
                        else:
                            bg.paste(img_resized.convert('RGBA'), mask=img_resized.convert('RGBA').split()[3])
                        bg.save(file_path, "JPEG")
                    else:
                        img_resized.convert('RGB').save(file_path, "JPEG")
            except Exception as e:
                print(f" Skipped bad/corrupted image: {filename} in {folder} -> {e}")

print(" All set! Images are cleaned, converted, and standardized.\n")
print(" Step 2: Starting the 5x augmentation pipeline...")

# --- STEP 2: AUGMENTATION LOOP ---
for folder in folders:
    full_folder_path = os.path.join(dataset_path, folder)
    
    # Count original images in current plant folder (ignore nested folders)
    original_images = [f for f in os.listdir(full_folder_path) if os.path.isfile(os.path.join(full_folder_path, f))]
    num_original_images = len(original_images)
    
    if num_original_images == 0:
        print(f" Folder '{folder}' is empty! Skipping this one.")
        continue
        
    print(f" Processing: {folder} ({num_original_images} original images)")
    
    # Init Augmentor pipeline (it creates an 'output' folder inside this plant directory automatically)
    p = Augmentor.Pipeline(full_folder_path)
    
    # Real-world field condition tweaks (mimicking outdoor hiking angles & lighting)
    p.rotate_random_90(probability=0.5)                   # Random 90-degree turns
    p.flip_left_right(probability=0.5)                     # Mirror flip
    p.flip_top_bottom(probability=0.3)                     # Upside-down flip
    p.random_brightness(probability=0.7, min_factor=0.8, max_factor=1) # Sun/shade lighting
    p.random_color(probability=0.4, min_factor=0.8, max_factor=1.2)      # Slight camera color differences
    p.random_contrast(probability=0.5, min_factor=0.8, max_factor=1.2)   # High/low daylight contrast
    
    # Generate 5 augmented copies for every single image
    target_samples = num_original_images * 5
    p.sample(target_samples)
    
    print(f" └─ ✅ Done! Generated 5x images inside '{folder}/output'")

print("\n All done, All 51 plant categories now have 5x augmented copies ready.")