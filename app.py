import streamlit as st
import joblib
import tempfile
import numpy as np
import pandas as pd
from extract_features import extract_features

st.set_page_config(page_title="Emotion Detection", layout="centered")

st.title("🎤 Human Emotion Detection from Voice")
st.write("Upload a WAV file OR record your voice to detect emotion.")

model = joblib.load("model.pkl")

# ===============================
# SESSION STORAGE (max 5 tries)
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []

MAX_ATTEMPTS = 5

# ===============================
# AUDIO INPUT
# ===============================
if len(st.session_state.history) < MAX_ATTEMPTS:

    uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])
    audio_recorded = st.audio_input("Or record your voice")

    audio_source = None

    if uploaded_file is not None:
        audio_source = uploaded_file
    elif audio_recorded is not None:
        audio_source = audio_recorded

    if audio_source is not None:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            temp.write(audio_source.read())
            temp_path = temp.name

        try:
            features = extract_features(temp_path)
            features = np.expand_dims(features, axis=0)

            prediction = model.predict(features)[0]
            probs = model.predict_proba(features)[0]
            confidence = np.max(probs) * 100

            st.success(f"Predicted Emotion: **{prediction.upper()}**")
            st.info(f"Model Confidence: {confidence:.2f}%")

            # Save attempt
            st.session_state.history.append({
                "Attempt": len(st.session_state.history) + 1,
                "Emotion": prediction,
                "Confidence (%)": round(confidence, 2)
            })

        except Exception as e:
            st.error(f"Error processing audio: {e}")

else:
    st.warning("Maximum 5 attempts reached for this session.")

# ===============================
# SHOW SESSION RESULTS
# ===============================
if st.session_state.history:

    st.subheader("Session Attempts (Max 5)")
    df = pd.DataFrame(st.session_state.history)
    st.table(df)

    summary = df["Emotion"].value_counts()
    st.subheader("Session Emotion Summary")
    st.bar_chart(summary)

# ===============================
# RESET BUTTON
# ===============================
if st.button("🔄 Reset Session"):
    st.session_state.history = []
    st.rerun()
