import os
import PIL
from PIL import Image
import warnings

# This is Path to my training data:
DATA_DIR = "03_Data/train"

def verify_images(folder_path):
    print(f" Starting verification in: {folder_path}")
    print("="*50)
    
    deleted_count = 0
    total_scanned = 0

    # Loop through each plant folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                total_scanned += 1
                file_path = os.path.join(root, file)
                
                try:
                    # Try to open and verify the image
                    img = Image.open(file_path)
                    img.verify() # Verify that this is an actual image
                except (IOError, SyntaxError, PIL.UnidentifiedImageError) as e:
                    print(f" Corrupt file found and deleted: {file_path}")
                    os.remove(file_path)
                    deleted_count += 1
    
    print("="*50)
    print(f" Verification Complete!")
    print(f"Total Scanned: {total_scanned}")
    print(f"Total Deleted: {deleted_count}")

if __name__ == "__main__":
    # First, verify training data
    verify_images(DATA_DIR)