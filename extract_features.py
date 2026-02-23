import librosa
import numpy as np

def extract_features(file_path):

    # Load audio
    y, sr = librosa.load(file_path, sr=22050)

    # Normalize audio
    y = librosa.util.normalize(y)

    # Fix length to 3 seconds
    max_len = 22050 * 3
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    else:
        y = y[:max_len]

    # ===============================
    # MFCC (mean + std)
    # ===============================
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # ===============================
    # Chroma
    # ===============================
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # ===============================
    # Spectral Contrast
    # ===============================
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)

    # ===============================
    # Zero Crossing Rate
    # ===============================
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    # ===============================
    # RMS Energy
    # ===============================
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)

    return np.hstack([
        mfcc_mean,
        mfcc_std,
        chroma_mean,
        contrast_mean,
        zcr_mean,
        rms_mean
    ])
