import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# 1. LOAD THE V3 BRAIN 
V3_MODEL_PATH = "04_Models/caliplant_v3_51classes.keras"
if not os.path.exists(V3_MODEL_PATH):
    print(" Error: V3 model not found. Run the V3 script first!")
    exit()

model = tf.keras.models.load_model(V3_MODEL_PATH)

# 2. UNFREEZE THE DEEP LAYERS 
# We access the MobileNetV2 base (which is the first layer of our Sequential model)
base_model = model.layers[0]
base_model.trainable = True

# We only want to fine-tune the top layers, not the very bottom ones.
# MobileNetV2 has ~154 layers. We keep the first 100 frozen.
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# 3. RE-COMPILE WITH MICRO-LEARNING RATE 
# Critical: If the learning rate is high, it will 'break' the pre-trained weights.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. PREPARE DATA 
DATA_DIR = "03_Data"
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'validation'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# 5. EXECUTE THE UPGRADE
print("\n Fine-tuning V4... this is where the model becomes an expert.")
epochs_fine = 10 # 10 additional rounds of deep learning

model.fit(
    train_generator,
    epochs=epochs_fine,
    validation_data=val_generator
)

# 6. SAVE AS VERSION 4
model.save("04_Models/caliplant_v4_finetuned.keras")
print("\n SUCCESS: Version 4 saved successfully.")