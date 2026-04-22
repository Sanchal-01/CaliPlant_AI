import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# 1. LOAD BRAIN 
V4_MODEL_PATH = "04_Models/caliplant_v4_finetuned.keras"
if not os.path.exists(V4_MODEL_PATH):
    print("Error: V4 model nahi mila! Pehle V4 complete karo.")
    exit()

model = tf.keras.models.load_model(V4_MODEL_PATH)

#2. DEEP UNFREEZING (Level 50)
base_model = model.layers[0]
base_model.trainable = True

# Hum 154 mein se 100 layers unfreeze kar rahe hain = ab model bahut deep seekhega
fine_tune_at = 50 
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# 3. SMART COMPILATION
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005), # Ultra-small LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. SMART CALLBACKS (The Secret Sauce) 
# Accuracy nahi badhi toh training rok do
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Agar accuracy stuck ho jaye, toh learning rate aur kam kar do
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)

# Sirf BEST wala version hi save karna
checkpoint = ModelCheckpoint("04_Models/caliplant_v5_best.keras", monitor='val_accuracy', save_best_only=True)

#5. DATA (With Heavy Augmentation)
DATA_DIR = "03_Data"
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(os.path.join(DATA_DIR, 'train'), target_size=(224, 224), batch_size=32, class_mode='categorical')
val_generator = val_datagen.flow_from_directory(os.path.join(DATA_DIR, 'validation'), target_size=(224, 224), batch_size=32, class_mode='categorical')

# 6. EXECUTE V5
print("\nV5: Push to the Limit! Even if the training is slow, reults would be marvelous.")
model.fit(
    train_generator,
    epochs=20, # Zyada epochs kyunki EarlyStopping handle kar lega
    validation_data=val_generator,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

print("\n V5 Training Complete! Best version saved in 04_Models.")