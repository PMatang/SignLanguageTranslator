import cv2
import os
import time

# --- Config ---
SIGN_LABEL = "Okay"          # Change this to B, C, etc. for each letter
NUM_IMAGES = 500
SAVE_DIR = f"data/raw/{SIGN_LABEL}"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0
collecting = False

print(f"Collecting data for sign: {SIGN_LABEL}")
print("Press SPACE to start collecting. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    cv2.putText(display, f"Sign: {SIGN_LABEL} | Saved: {count}/{NUM_IMAGES}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if collecting:
        cv2.putText(display, "COLLECTING...", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        filename = os.path.join(SAVE_DIR, f"{count}.jpg")
        cv2.imwrite(filename, frame)
        count += 1
        time.sleep(0.05)  # 20 FPS collection speed

        if count >= NUM_IMAGES:
            print(f"Done! {NUM_IMAGES} images saved for '{SIGN_LABEL}'.")
            break

    cv2.imshow("Data Collector", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        collecting = True
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()