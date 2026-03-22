import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


# 1. Loading the trained brain or say (Model):
model_path = '04_Models/plant_model_v1.h5'
if not os.path.exists(model_path):
    print(f"❌ Error: Model file not found at {model_path}")
    exit()

model = tf.keras.models.load_model(model_path)

# Classes ke naam (usi order mein jaise folders hain): 
class_names = ['California Poppy', 'California Sagebrush', 'Coast Live Oak', 'Joshua Tree', 'Poison Oak']

def predict_my_plant(img_path):
    if not os.path.exists(img_path):
        print(f"Error: Image file not found at {img_path}")
        return

    # 2. Image Preprocessing (AI ke size 150x150 aur scale mein convert karna)
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # 3. Prediction (AI ka guess)
    prediction = model.predict(img_array)
    
    # 4. Find the best guess
    score = tf.nn.softmax(prediction[0]) # Convert the prediction values to probabilities
    result_index = np.argmax(score)
    result_label = class_names[result_index]
    confidence = 100 * np.max(score)

    print(f"\n-----------------------------------")
    print(f"Testing Image: {os.path.basename(img_path)}")
    print(f"AI Guess: '{result_label}'")
    print(f"Confidence Score: {confidence:.2f}%")
    print(f"-----------------------------------\n")

# EXECUTION (Test with a few photos)
# Path of photos :  
predict_my_plant("03_Data/test_photos/image04.jpg")
predict_my_plant("03_Data/test_photos/images03.jpg")
predict_my_plant("03_Data/test_photos/images5.jpg")
predict_my_plant("03_Data/test_photos/img1.jpg")