"""
predict.py — Quick CLI prediction on a single leaf image
==========================================================
Usage:
    python predict.py path/to/leaf.jpg

Requires trained model at  model/plant_disease_model.h5
"""

import sys, os
import numpy as np

CLASSES = [
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Potato Late Blight",
    "Corn Common Rust",
    "Healthy",
]

BAR_WIDTH = 30


def predict_single(image_path: str):
    try:
        import tensorflow as tf
        from PIL import Image
    except ImportError as e:
        print(f"❌  Missing dependency: {e}")
        sys.exit(1)

    model_path = os.path.join(os.path.dirname(__file__), "model", "plant_disease_model.h5")
    if not os.path.exists(model_path):
        print(f"❌  Model not found at {model_path}")
        print("    Train first:  python model/train_model.py")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"❌  Image not found: {image_path}")
        sys.exit(1)

    print(f"🔄  Loading model…")
    model = tf.keras.models.load_model(model_path)

    print(f"🔬  Processing image: {image_path}")
    img = Image.open(image_path).convert("RGB").resize((128, 128))
    arr = np.array(img, dtype=np.float32) / 255.0
    inp = np.expand_dims(arr, 0)

    print(f"🤖  Running inference…\n")
    preds = model.predict(inp, verbose=0)[0]

    idx  = int(np.argmax(preds))
    conf = float(preds[idx]) * 100

    print("=" * 50)
    print(f"  🌿 PREDICTION: {CLASSES[idx]}")
    print(f"  📊 Confidence: {conf:.1f}%")
    print("=" * 50)
    print()

    print("All class scores:")
    for i, (cls, sc) in enumerate(zip(CLASSES, preds)):
        pct   = float(sc) * 100
        filled = int(pct / 100 * BAR_WIDTH)
        bar   = "█" * filled + "░" * (BAR_WIDTH - filled)
        marker = " ◄" if i == idx else ""
        print(f"  {cls:<30}  [{bar}]  {pct:5.1f}%{marker}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    predict_single(sys.argv[1])
