import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# 1. SETTINGS 
MODEL_PATH = "04_Models/caliplant_v5_best.keras"
DATA_DIR = "03_Data/train"  # Used to get the 51 plant names

# Load class names dynamically from our folders
if not os.path.exists(DATA_DIR):
    print(f" Error: Data directory '{DATA_DIR}' not found!")
    exit()

class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

# 2. LOAD BEST BRAIN 
print(f" Loading V5 Expert Model: {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    print("Error: V5 model file not found. Check the path!")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)

def predict_plant(img_path):
    # Image Preprocessing
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make Prediction
    predictions = model.predict(img_array)[0]
    
    # Get EXACTLY the top 2 highest indices
    # We sort all, take the last 2, and reverse them to get Rank 1 first
    top_2_indices = np.argsort(predictions)[-2:][::-1]
    
    print("\n" + "="*40)
    print(f"IMAGE: {os.path.basename(img_path)}")
    print("="*40)
    
    for i, idx in enumerate(top_2_indices):
        label = class_names[idx]
        conf_score = predictions[idx] * 100
        
        rank_label = "🥇 TOP" if i == 0 else "🥈 2ND"
        
        # Use a higher multiplier for the bar if you want them to look longer
        bar_visual = "█" * int(conf_score / 2) 
        
        print(f"{rank_label}: {label:25} | {conf_score:6.2f}% {bar_visual}")
    
    print("="*40 + "\n")
    
if __name__ == "__main__":
    # Change this to the path of the photo you want to test
    TEST_IMAGE = "03_Data/test_photos/test_image_06.jpg" 
    
    if os.path.exists(TEST_IMAGE):
        predict_plant(TEST_IMAGE)
    else:
        # If the file doesn't exist, show instructions
        print(f"Please put a photo in: {TEST_IMAGE}")
        print("Tip: Use any image from your 'validation' folder to test it.")