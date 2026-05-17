# Run this in SecuSign folder to install venv
python -m venv venv

# run this in your terminal to activate venv
# For Windows
venv\Scripts\activate

# For Mac/Linux
source venv/bin/activate

# to insatll all the dependencies in your venv
pip install -r requirements.txt

# From inside your SecuSign/ folder with venv active:

# Phase 1 — collect data (change SIGN_LABEL in the script for each letter)
python src/phase1_collect.py

# Phase 2 — clean the images
python src/phase2_dip.py

# Phase 3 — extract landmarks to CSV
python src/phase3_landmarks.py

# Phase 4 — train the model (do 4a OR 4b, or both)
python src/phase4_train_cnn.py
python src/phase4_train_lstm.py

# Phase 5 — launch the app
streamlit run src/phase5_app.py