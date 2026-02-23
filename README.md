**Human Emotion Detection From Voice**

A machine learning web application that detects human emotions from speech audio.
The system analyzes voice recordings and predicts emotions such as happy, sad, anger, fearful, neutral, disgust, and surprise using extracted acoustic features.
This project demonstrates an end-to-end ML pipeline from audio preprocessing to real-time prediction through a web interface.

**Project Overview**:

A machine learning web application that detects human emotions from speech audio using acoustic feature analysis.
The system processes voice recordings and predicts emotional states such as:

1. Happy
2. Sad
3. Angry
4. Fearful 
5. Neutral 
6. Disgust 
7. Surprise

This project demonstrates a complete end-to-end ML pipeline including data preprocessing, feature extraction, model training, evaluation, and real-time prediction through a web interface.

**Project Motivation**:

Human emotions influence vocal characteristics such as pitch, tone, rhythm, and energy.
By analyzing these speech properties using signal processing and machine learning techniques, emotional states can be predicted automatically.

This project applies classical machine learning methods to speech emotion recognition and provides a practical web-based application for real-time emotion prediction.

The project reflects hands-on practice in:

✔️Feature normalization

✔️Encoding and preprocessing

✔️Model training and evaluation

✔️ Debugging ML workflows

✔️Deployment-ready application design

**System Architecture:**

The system follows a standard machine learning workflow:

Audio Input → Preprocessing → Feature Extraction → Model Prediction → Result Display

**Dataset**:

The model was trained using the RAVDESS Speech Emotion Dataset, which contains labeled emotional speech recordings.

Dataset file Path: "C:\Users\rasag\OneDrive\Documents\RAVDNESS"

Dataset characteristics:

1. Multiple speakers

2. Balanced emotion classes

3. High-quality WAV audio

4.Emotion labels mapped to classification categories

**Methodology**:

1. **Audio Preprocessing**

To ensure consistent input for machine learning, the following preprocessing steps were applied:

1, Audio loaded at fixed sampling rate

2, Amplitude normalization

3, Trimmed to uniform duration (3 seconds)

4, Noise and signal inconsistencies reduced

5, Converted into structured numerical form


2.** Feature Extraction**

Multiple acoustic features were extracted from each audio signal:

1, FCC (Mel Frequency Cepstral Coefficients)

Represents speech frequency characteristics similar to human hearing.

2, Chroma Features

Capture pitch class information from audio signals.

3, Spectral Contrast

Measures energy differences between frequency bands.

4, Zero Crossing Rate

Indicates signal noisiness and frequency variation.

5, RMS Energy

Represents loudness intensity of the signal.


3. **Model Training**

Machine learning pipeline components:

1, Stratified train-test split (80–20)

2, Feature scaling for normalization

3, Extra Trees Classifier

4, Model evaluation using classification metrics

5, Trained model saved for future predictions


4.** Model Evaluation**

Evaluation metrics used:

1, Accuracy Score

2, Precision

3, Recall

4, F1-score

5, Classification Report

**Performance Result**:

Overall model accuracy: ~67%

**Web Application Interface**:

An interactive web application was developed to enable real-time emotion detection from speech.

User Capabilities

✔️Upload WAV audio file

✔️Record voice directly

✔️View predicted emotion

✔️See prediction confidence score

✔️Track session prediction history

✔️Visualize emotion distribution

**Technologies Used**:

Programming & Libraries:

1. Python

2. NumPy

3. Librosa

4. Scikit-learn

5. Joblib

Application Framework:

6. Streamlit

Machine Learning Techniques:

8. Feature normalization

9. Statistical feature engineering

10. Ensemble classification
