import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import csv
import os
import urllib.request

PROCESSED_DIR = "data/raw"
LANDMARKS_DIR = "data/landmarks"
MODEL_PATH = "hand_landmarker.task"
os.makedirs(LANDMARKS_DIR, exist_ok=True)

# --- Download the hand landmarker model if not present ---
if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe hand landmark model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Download complete.")

# --- Build the landmarker ---
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE
)

def extract_landmarks(image_path, landmarker):
    """Returns a flat list of 63 values (21 points x,y,z) or None."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    coords = []
    for lm in result.hand_landmarks[0]:
        coords.extend([lm.x, lm.y, lm.z])

    return coords  # 63 values

def build_dataset():
    output_csv = os.path.join(LANDMARKS_DIR, "dataset.csv")

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            header = ["label"] + [f"x{i}" for i in range(63)]
            writer.writerow(header)

            for label in sorted(os.listdir(PROCESSED_DIR)):
                folder = os.path.join(PROCESSED_DIR, label)
                if not os.path.isdir(folder):
                    continue

                images = [fi for fi in os.listdir(folder) if fi.lower().endswith(".jpg")]
                print(f"Extracting landmarks for '{label}': {len(images)} images...")
                success = 0

                for fname in images:
                    path = os.path.join(folder, fname)
                    coords = extract_landmarks(path, landmarker)
                    if coords is not None:
                        writer.writerow([label] + coords)
                        success += 1

                print(f"  Saved {success}/{len(images)} valid rows for '{label}'.")

    print(f"\nDataset saved to: {output_csv}")

if __name__ == "__main__":
    build_dataset()
