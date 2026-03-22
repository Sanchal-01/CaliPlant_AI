import os  # Folder management ke liye zaruri hai
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. DATA PREPARATION (Augmentation)
# Logic: Making new images from old ones to help AI learn better.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
# Explanation: Isme hum "Rules" set kar rahe hain. rescale pixels ko chota karta hai taaki calculation fast ho.
# rotation aur flip se AI ko har baar thodi tedhi-medhi photo dikhegi taaki wo har angle se pehchanna seekhe.





val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '03_Data/train',
    target_size=(150, 150),
    batch_size=4,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    '03_Data/validation',
    target_size=(150, 150),
    batch_size=4,
    class_mode='categorical'
)

# 2. MODEL BUILDING (CNN Structure)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(5, activation='softmax')
])
# Explanation: Ye AI ka "Brain Structure" hai. Conv2D filter ki tarah edges aur colors dhoondta hai. MaxPooling sirf zaruri details rakhta hai. 
# Aakhir mein Dense(5) humein 5 plants ke hisaab se result deta hai.



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. TRAINING
print("\n Training shuru ho rahi hai...")
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# 4. SAVE MODEL
# Creating a folder to save our trained 'Brain' (Model).
if not os.path.exists('04_Models'):
    os.makedirs('04_Models')

model.save('04_Models/plant_model_v1.h5')
print("\n Model save ho gaya: 04_Models/plant_model_v1.h5")
