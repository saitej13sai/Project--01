# 🌿 Plant Disease Detection from Leaf Images

> CNN-based plant disease classifier | TensorFlow · Keras · OpenCV · Streamlit

---

## 📁 Project Structure

```
plant_disease_detection/
│
├── app/
│   └── app.py                      ← Streamlit GUI (drag & drop + diagnosis)
│
├── model/
│   ├── train_model.py              ← CNN training (4 conv blocks + augmentation)
│   ├── plant_disease_model.h5      ← Saved model (generated after training)
│   ├── training_curves.png         ← Acc/Loss plots (generated after training)
│   └── confusion_matrix.png        ← Confusion matrix (generated after training)
│
├── utils/
│   └── preprocess.py               ← OpenCV: CLAHE, denoise, leaf segmentation
│
├── dataset_sample/
│   ├── download_dataset.py         ← Auto-download via Kaggle API
│   ├── Tomato_Bacterial_spot/      ← Place images here
│   ├── Tomato_Early_blight/
│   ├── Potato_Late_blight/
│   ├── Corn_Common_rust/
│   └── Healthy/
│
├── predict.py                      ← CLI: predict a single image
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Get the PlantVillage dataset

**Option A — Kaggle API (automated)**
```bash
# 1. Get your API token from https://www.kaggle.com/settings
# 2. Save as ~/.kaggle/kaggle.json
# 3. Run:
python dataset_sample/download_dataset.py
```

**Option B — Manual download**
1. Visit https://www.kaggle.com/datasets/emmarex/plantdisease
2. Download & extract the ZIP
3. Copy these 5 folders into `dataset_sample/`:

```
dataset_sample/
├── Tomato_Bacterial_spot/    (rename from Tomato___Bacterial_spot)
├── Tomato_Early_blight/      (rename from Tomato___Early_blight)
├── Potato_Late_blight/       (rename from Potato___Late_blight)
├── Corn_Common_rust/         (rename from Corn_(maize)___Common_rust_)
└── Healthy/                  (copy from Tomato___healthy)
```

### Step 3 — Train the CNN
```bash
python model/train_model.py
```
- Trains for up to 25 epochs with early stopping
- Saves `model/plant_disease_model.h5`
- Saves accuracy/loss curves and confusion matrix
- Expected validation accuracy: **~90–95%**

### Step 4 — Run the Streamlit app
```bash
streamlit run app/app.py
```
Open **http://localhost:8501**

### Step 5 (optional) — Predict from CLI
```bash
python predict.py path/to/leaf.jpg
```

---

## 🧠 CNN Architecture

```
Input: 128 × 128 × 3

Conv2D(32) → BN → Conv2D(32) → MaxPool(2) → Dropout(0.25)
Conv2D(64) → BN → Conv2D(64) → MaxPool(2) → Dropout(0.25)
Conv2D(128) → BN → Conv2D(128) → MaxPool(2) → Dropout(0.40)
Conv2D(256) → BN → MaxPool(2) → Dropout(0.40)

GlobalAveragePooling2D
Dense(256, relu) → BN → Dropout(0.5)
Dense(5, softmax)
```

**Optimizer:** Adam (lr=1e-3, ReduceLROnPlateau)
**Loss:** Categorical Crossentropy
**Augmentation:** rotation, shift, zoom, flip, brightness

---

## 🌱 Detectable Classes

| # | Class | Crop | Severity | Pathogen |
|---|-------|------|----------|----------|
| 0 | Tomato Bacterial Spot | Tomato | Moderate | Xanthomonas bacteria |
| 1 | Tomato Early Blight | Tomato | High | Alternaria solani |
| 2 | Potato Late Blight | Potato | Critical | Phytophthora infestans |
| 3 | Corn Common Rust | Corn | Moderate | Puccinia sorghi |
| 4 | Healthy | All | None | — |

---

## 🖥️ GUI Features

- Drag-and-drop image upload
- Real-time CNN inference
- Confidence bar for all 5 classes
- OpenCV leaf segmentation preview
- Diagnosis report: cause, symptoms, severity
- Treatment & prevention action plan

---

## ☁️ Deploy to Streamlit Cloud (Free)

```
1. Push this folder to GitHub
2. Go to https://share.streamlit.io
3. New app → select repo → main file: app/app.py
4. Deploy → live in 2 minutes
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| TensorFlow / Keras | CNN model build & train |
| OpenCV | CLAHE, denoise, leaf segmentation |
| Streamlit | Interactive web GUI |
| NumPy / Pillow | Array & image handling |
| Matplotlib / Seaborn | Training plots, confusion matrix |
| scikit-learn | Classification report, metrics |

---

## 📦 Deliverables Checklist

- [x] Trained model (`model/plant_disease_model.h5`)
- [x] GUI app (`app/app.py`)
- [x] Training script (`model/train_model.py`)
- [x] Preprocessing utils (`utils/preprocess.py`)
- [x] Dataset download helper (`dataset_sample/download_dataset.py`)
- [x] CLI predictor (`predict.py`)
- [x] requirements.txt
- [x] README with setup instructions

---

## 📝 License
MIT — free for academic and personal use.
