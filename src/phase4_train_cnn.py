import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = "data/processed"
MODEL_SAVE = "models/cnn_model.h5"
os.makedirs("models", exist_ok=True)

# --- Data loaders ---
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=False
)

train_data = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, color_mode="grayscale",
    batch_size=BATCH_SIZE, class_mode="categorical", subset="training"
)

val_data = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, color_mode="grayscale",
    batch_size=BATCH_SIZE, class_mode="categorical", subset="validation"
)

NUM_CLASSES = len(train_data.class_indices)
print(f"Classes found: {train_data.class_indices}")

# --- CNN Architecture ---
model = models.Sequential([
    layers.Input(shape=(64, 64, 1)),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --- Training ---
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

model.save(MODEL_SAVE)
print(f"\nModel saved to {MODEL_SAVE}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]*100:.1f}%")