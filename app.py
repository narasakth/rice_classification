import os
import io
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify

# ── App ─────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Configuration ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model_resnetv2.pth")

CLASS_NAMES = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
IMG_SIZE = 224

# ✅ HF Spaces: CPU only
DEVICE = torch.device("cpu")

# ── Image preprocessing ────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ── Load model (ครั้งเดียว) ────────────────────────────────────
def load_model():
    print("🔄 Loading model...")

    # สร้าง architecture ให้ตรง (เลือกอันเดียว ลด RAM)
    model = timm.create_model(
        "resnetv2_50",
        pretrained=False,
        num_classes=len(CLASS_NAMES)
    )

    state_dict = torch.load(MODEL_PATH, map_location="cpu")

    # กรณี save แบบมี key "state_dict"
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # ลบ prefix model. ถ้ามี
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("model.", "", 1)] = v

    model.load_state_dict(cleaned, strict=False)
    model.eval()
    model.to(DEVICE)

    print("✅ Model loaded successfully")
    return model


model = load_model()

# ── Routes ─────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        results = [
            {
                "class": CLASS_NAMES[i],
                "confidence": round(probs[i].item() * 100, 2)
            }
            for i in range(len(CLASS_NAMES))
        ]

        results.sort(key=lambda x: x["confidence"], reverse=True)

        return jsonify({
            "success": True,
            "prediction": results[0]["class"],
            "confidence": results[0]["confidence"],
            "all_predictions": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    