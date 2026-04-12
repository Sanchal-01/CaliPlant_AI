import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Running this from the root folder
DATA_DIR = "03_Data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15                  # 15 passes over the data to ensure 51 classes are learned well this I think is important.

#1. DATA AUGMENTATION _This creates variations of our images so the AI doesn't just memorize pixels.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation data should only be rescaled, not changed
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'validation'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Extracting the 51 class names automatically from the folders
class_names = sorted(list(train_generator.class_indices.keys()))
print(f" Total Classes detected: {len(class_names)}")

# 2. BUILDING THE V3 MODEL (Transfer Learning) ka use kiya hai.
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), 
    include_top=False, 
    weights='imagenet'
)
base_model.trainable = False  # Keep the "general knowledge" frozen : Initial Layers (0-50),Middle Layers (50-100) & Final Layers (100-154)

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),                        # Increased neurons for 51 classes
    layers.Dropout(0.4),                                         # Prevents overfitting
    layers.Dense(len(class_names), activation='softmax')         # The 51-way switch
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 3. EXECUTION 
print("\n Professor's Note: Training has started requires patience..")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# 4. SAVING THE VERSION 3 MODEL 
if not os.path.exists("04_Models"):
    os.makedirs("04_Models")

model.save("04_Models/caliplant_v3_51classes.keras")
print("\n SUCCESS: Model saved as '04_Models/caliplant_v3_51classes.keras'")