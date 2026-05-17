import os
import cv2
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
import time
import urllib.request

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="SecuSign", layout="wide", page_icon="🤟")

st.markdown("""
    <style>
        .main-title  { font-size: 2rem; font-weight: 700; margin-bottom: 0; }
        .sub-title   { font-size: 1rem; color: grey; margin-bottom: 1.5rem; }
        .sign-box    { font-size: 3rem; font-weight: 800; text-align: center; padding: 0.5rem; }
        .sentence-box{ background: #f0f2f6; color:black; border-radius: 10px; padding: 1rem;
                       font-size: 1.2rem; min-height: 60px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🤟 SecuSign</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Real-time Sign Language to Audio Translator</p>', unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
LANDMARK_MODEL_PATH = "hand_landmarker.task"
CNN_MODEL_PATH      = "models/cnn_model.h5"
LSTM_MODEL_PATH     = "models/lstm_model.h5"
LABEL_PATH          = "models/label_classes.npy"

# ── Download MediaPipe hand landmarker model if missing ────────────────────────
@st.cache_resource
def download_mp_model():
    if not os.path.exists(LANDMARK_MODEL_PATH):
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
        with st.spinner("Downloading MediaPipe hand model (first run only)..."):
            urllib.request.urlretrieve(url, LANDMARK_MODEL_PATH)
    return LANDMARK_MODEL_PATH

download_mp_model()

# ── Load AI model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_ai_model():
    """Load whichever model exists. LSTM takes priority over CNN."""
    if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LABEL_PATH):
        model  = tf.keras.models.load_model(LSTM_MODEL_PATH)
        labels = np.load(LABEL_PATH, allow_pickle=True)
        return model, labels, "LSTM"

    if os.path.exists(CNN_MODEL_PATH) and os.path.exists(LABEL_PATH):
        model  = tf.keras.models.load_model(CNN_MODEL_PATH)
        labels = np.load(LABEL_PATH, allow_pickle=True)
        return model, labels, "CNN"

    return None, None, None

model, CLASSES, MODEL_TYPE = load_ai_model()

# ── Build MediaPipe landmarker ─────────────────────────────────────────────────
@st.cache_resource
def build_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=LANDMARK_MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.HandLandmarker.create_from_options(options)

landmarker = build_landmarker()

# ── DIP preprocessing ──────────────────────────────────────────────────────────
def dip_preprocess(frame):
    """Gaussian blur → Otsu binarization → Canny edges combined into one mask."""
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Otsu binarization
    _, otsu = cv2.threshold(blurred, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Canny edges
    edges = cv2.Canny(blurred, 50, 150)

    # Skin segmentation in HSV
    hsv        = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0,  20,  70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask  = cv2.inRange(hsv, lower_skin, upper_skin)

    # Morphological cleanup
    kernel    = np.ones((3, 3), np.uint8)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    skin_mask = cv2.erode(skin_mask,  kernel, iterations=1)

    # Combine: show whichever has signal
    combined = cv2.bitwise_or(otsu, edges)
    combined = cv2.bitwise_or(combined, skin_mask)

    return combined

# ── Landmark extraction ────────────────────────────────────────────────────────
def get_landmarks(frame):
    """Returns (coords_array or None, annotated_frame)."""
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = landmarker.detect(mp_image)

    annotated = frame.copy()

    if not result.hand_landmarks:
        return None, annotated

    # Draw connections manually (new API does not have drawing_utils)
    h, w = frame.shape[:2]
    connections = [
        (0,1),(1,2),(2,3),(3,4),          # thumb
        (0,5),(5,6),(6,7),(7,8),          # index
        (0,9),(9,10),(10,11),(11,12),     # middle
        (0,13),(13,14),(14,15),(15,16),   # ring
        (0,17),(17,18),(18,19),(19,20),   # pinky
        (5,9),(9,13),(13,17),             # palm
    ]
    pts = [(int(lm.x * w), int(lm.y * h))
           for lm in result.hand_landmarks[0]]

    for a, b in connections:
        cv2.line(annotated, pts[a], pts[b], (0, 200, 100), 2)
    for x, y in pts:
        cv2.circle(annotated, (x, y), 5, (255, 255, 255), -1)
        cv2.circle(annotated, (x, y), 5, (0, 150, 80),   1)

    # Flat coordinate array
    coords = []
    for lm in result.hand_landmarks[0]:
        coords.extend([lm.x, lm.y, lm.z])

    return np.array(coords, dtype="float32"), annotated

# ── TTS (runs in background thread) ───────────────────────────────────────────
def speak(text):
    def _run():
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass   # TTS failure should never crash the app
    threading.Thread(target=_run, daemon=True).start()

# ── Predict from landmark coords ───────────────────────────────────────────────
def predict(coords):
    if model is None or coords is None:
        return "—", 0.0

    # Shape depends on model type
    if MODEL_TYPE == "LSTM":
        inp = coords.reshape(1, 1, 63)
    else:
        # CNN expects image — landmark-based CNN still uses (1,1,63) reshape
        inp = coords.reshape(1, 1, 63)

    preds      = model.predict(inp, verbose=0)[0]
    idx        = int(np.argmax(preds))
    label      = str(CLASSES[idx])
    confidence = float(preds[idx])
    return label, confidence

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    confidence_threshold = st.slider(
        "Confidence threshold", 0.50, 0.99, 0.85, 0.01,
        help="Minimum confidence to accept a prediction."
    )
    cooldown_sec = st.slider(
        "Word cooldown (seconds)", 0.5, 5.0, 2.0, 0.5,
        help="Wait time before the same word is spoken again."
    )
    enable_tts = st.toggle("🔊 Speak predictions", value=True)
    clear_btn  = st.button("🗑️ Clear sentence")

    st.divider()
    st.markdown("**Model loaded:**")
    if MODEL_TYPE:
        st.success(f"{MODEL_TYPE} — `{LSTM_MODEL_PATH if MODEL_TYPE=='LSTM' else CNN_MODEL_PATH}`")
    else:
        st.warning("No trained model found.\nTrain one first with phase4_train_*.py")

    st.divider()
    st.markdown("**How to use**")
    st.markdown("""
1. Toggle **Start camera** below.  
2. Hold a sign in front of your webcam.  
3. The app detects, translates, and speaks.  
4. Words are added to the sentence strip.  
    """)

# ── Session state ──────────────────────────────────────────────────────────────
if "sentence"       not in st.session_state: st.session_state.sentence       = []
if "last_spoken"    not in st.session_state: st.session_state.last_spoken    = ""
if "last_speak_time"not in st.session_state: st.session_state.last_speak_time= 0.0

if clear_btn:
    st.session_state.sentence        = []
    st.session_state.last_spoken     = ""
    st.session_state.last_speak_time = 0.0

# ── Main UI placeholders ───────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("#### 📷 Raw video")
    raw_ph = st.empty()

with col2:
    st.markdown("#### 🔬 DIP mask")
    mask_ph = st.empty()

with col3:
    st.markdown("#### 🤖 Landmark overlay")
    lm_ph = st.empty()

st.divider()

metric_col1, metric_col2, metric_col3 = st.columns(3)
sign_ph = metric_col1.empty()
conf_ph = metric_col2.empty()
sent_ph = st.empty()

run = st.toggle("▶️ Start camera", value=False, key="cam_toggle")

# ── Camera loop ────────────────────────────────────────────────────────────────
if run:
    if model is None:
        st.error("⚠️ No model found. Run `python src/phase4_train_lstm.py` first.")
        st.stop()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Cannot open webcam. Check that it is connected and not in use.")
        st.stop()

    while st.session_state.get("cam_toggle", False):
        ret, frame = cap.read()
        if not ret:
            st.error("Camera read failed.")
            break

        frame = cv2.flip(frame, 1)   # mirror view

        # --- DIP ---
        mask = dip_preprocess(frame)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        # --- Landmarks + annotated frame ---
        coords, annotated = get_landmarks(frame)

        # --- Prediction ---
        label, confidence = predict(coords)

        # --- Auto-speak logic ---
        now = time.time()
        if (
            enable_tts
            and coords is not None
            and confidence >= confidence_threshold
            and (
                label != st.session_state.last_spoken
                or now - st.session_state.last_speak_time > cooldown_sec
            )
        ):
            speak(label)
            st.session_state.sentence.append(label)
            st.session_state.last_spoken     = label
            st.session_state.last_speak_time = now

        # --- Display frames ---
        frame_rgb    = cv2.cvtColor(frame,     cv2.COLOR_BGR2RGB)
        annotated_rgb= cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        raw_ph.image(frame_rgb,     channels="RGB", use_container_width=True)
        mask_ph.image(mask_rgb,     channels="RGB", use_container_width=True)
        lm_ph.image(annotated_rgb,  channels="RGB", use_container_width=True)

        # --- Metrics ---
        sign_ph.metric("Detected sign", label if coords is not None else "—")
        conf_val = int(confidence * 100)
        conf_ph.progress(
            conf_val,
            text=f"Confidence: {conf_val}%  (threshold: {int(confidence_threshold*100)}%)"
        )

        # --- Sentence strip ---
        sentence_text = " ".join(st.session_state.sentence) if st.session_state.sentence else "_"
        sent_ph.markdown(
            f'<div class="sentence-box">📝 {sentence_text}</div>',
            unsafe_allow_html=True
        )

    cap.release()

else:
    # Placeholder images when camera is off
    raw_ph.info("Camera is off. Toggle **▶️ Start camera** to begin.")
    mask_ph.empty()
    lm_ph.empty()

    sentence_text = " ".join(st.session_state.sentence) if st.session_state.sentence else "_"
    sent_ph.markdown(
        f'<div class="sentence-box">📝 {sentence_text}</div>',
        unsafe_allow_html=True
    )
