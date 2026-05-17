import cv2
import os
import numpy as np

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def preprocess_image(image):
    """Apply DIP pipeline: Gaussian → Otsu → Canny."""
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Step 3: Otsu's binarization (auto threshold)
    _, otsu = cv2.threshold(blurred, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 4: Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Step 5: Skin color segmentation in HSV space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Step 6: Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    skin_mask = cv2.erode(skin_mask, kernel, iterations=1)

    return otsu, edges, skin_mask

def process_all():
    for label in os.listdir(RAW_DIR):
        input_folder = os.path.join(RAW_DIR, label)
        output_folder = os.path.join(PROCESSED_DIR, label)
        os.makedirs(output_folder, exist_ok=True)

        images = [f for f in os.listdir(input_folder) if f.endswith(".jpg")]
        print(f"Processing {len(images)} images for '{label}'...")

        for fname in images:
            img_path = os.path.join(input_folder, fname)
            image = cv2.imread(img_path)
            if image is None:
                continue

            otsu, edges, skin = preprocess_image(image)

            # Save the Otsu binarized version for CNN training
            save_path = os.path.join(output_folder, fname)
            cv2.imwrite(save_path, otsu)

        print(f"  Done: {label}")

    print("All images processed!")

if __name__ == "__main__":
    process_all()