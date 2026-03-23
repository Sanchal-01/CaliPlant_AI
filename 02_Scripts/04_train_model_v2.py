import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. DATA PREPARATION (Same as before in last train script on my scratch model training but with 160x160 size)
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('03_Data/train', target_size=(160, 160), batch_size=4, class_mode='categorical')
val_generator = val_datagen.flow_from_directory('03_Data/validation', target_size=(160, 160), batch_size=4, class_mode='categorical')

# 2. TRANSFER LEARNING SETUP
# 'include_top=False' matlab hum iska output layer hata kar apna 5-plant wala layer lagayenge
base_model = MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Iska purana "Gyan" freeze kar do

# 3. BUILDING CUSTOM HEAD
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(), # Data ko simplify karne ke liye
    layers.Dropout(0.2),             # Ratta-mar (Overfitting) rokne ke liye :)
    layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. TRAINING
print("\n Pro Training shuru ho rahi hai (MobileNetV2)...hihi")
model.fit(train_generator, epochs=10, validation_data=val_generator)

# 5. SAVE NEW MODEL
if not os.path.exists('04_Models'): os.makedirs('04_Models')
model.save('04_Models/plant_model_v2_pro.keras')                           # .keras extension is better! than .h5
print("\n Pro Model save ho gaya: 04_Models/plant_model_v2_pro.keras")