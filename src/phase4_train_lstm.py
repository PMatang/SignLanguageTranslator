import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import os

CSV_PATH = "data/landmarks/dataset.csv"
MODEL_SAVE = "models/lstm_model.h5"
SEQUENCE_LENGTH = 30   # frames per sign sequence
os.makedirs("models", exist_ok=True)

# --- Load data ---
df = pd.read_csv(CSV_PATH)
X = df.drop("label", axis=1).values.astype("float32")
y = df["label"].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)
NUM_CLASSES = y_cat.shape[1]

# Reshape into sequences: (samples, timesteps, features)
# Pad/truncate to SEQUENCE_LENGTH — here treating each row as one timestep
# For real sequences, group consecutive frames per sign
X_seq = X.reshape(-1, 1, 63)   # (N, 1, 63) — single frame as 1 timestep
# When you capture real video sequences, reshape to (N, SEQUENCE_LENGTH, 63)

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- LSTM Architecture ---
model = models.Sequential([
    layers.Input(shape=(X_seq.shape[1], 63)),

    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.3),

    layers.LSTM(128, return_sequences=False),
    layers.Dropout(0.3),

    layers.Dense(64, activation="relu"),
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
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32
)

model.save(MODEL_SAVE)
np.save("models/label_classes.npy", le.classes_)
print(f"Model saved → {MODEL_SAVE}")
print(f"Final accuracy: {history.history['val_accuracy'][-1]*100:.1f}%")