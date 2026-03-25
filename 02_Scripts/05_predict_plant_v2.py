import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# 1. Load the Pro Brain
model = tf.keras.models.load_model('04_Models/plant_model_v2_pro.keras')
class_names = ['California Poppy', 'California Sagebrush', 'Coast Live Oak', 'Joshua Tree', 'Poison Oak']

def predict_folder(folder_path):
    print(f"\n🔍 Scanning folder: {folder_path}")
    print("="*40)
    
    # Folder ki har file check karo
    for img_name in os.listdir(folder_path):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, img_name)
            
            # Preprocessing (MobileNetV2 uses 160x160)
            img = image.load_img(img_path, target_size=(160, 160))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Prediction
            preds = model.predict(img_array, verbose=0)
            result_index = np.argmax(preds)
            confidence = np.max(preds) * 100
            
            print(f"📷 Image: {img_name}")
            print(f"🌿 The Plantis likely to be: {class_names[result_index]} ({confidence:.2f}%)")
            print("-" * 20)

# --- CHALAO ---
predict_folder('03_Data/test_photos')