import os
import io
import json
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Configuration ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "best_model_resnetv2.pth")
CLASS_NAMES = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Image preprocessing pipeline ──────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Load model ─────────────────────────────────────────────────
def load_model():
    """Try multiple architectures to load the checkpoint."""
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    # Determine if checkpoint is a state_dict or a full model
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and any(k.startswith("model.") or k.startswith("stem.") or k.startswith("stages.") for k in checkpoint.keys()):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        # It might be a full model saved with torch.save(model, ...)
        model = checkpoint
        model = model.to(DEVICE)
        model.eval()
        return model

    # Clean up state dict keys (remove 'model.' prefix if present)
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k.replace("model.", "", 1) if k.startswith("model.") else k
        cleaned[new_key] = v

    # Try different timm ResNetV2 variants
    candidates = [
        "resnetv2_50",
        "resnetv2_101",
        "resnetv2_50d",
        "resnetv2_50x1_bit.goog_in21k_ft_in1k",
    ]

    for arch in candidates:
        try:
            model = timm.create_model(arch, pretrained=False, num_classes=len(CLASS_NAMES))
            model.load_state_dict(cleaned, strict=False)
            model = model.to(DEVICE)
            model.eval()
            print(f"✅ Model loaded successfully with architecture: {arch}")
            return model
        except Exception as e:
            print(f"⚠  {arch} failed: {e}")

    # Fallback: try torchvision resnet50 with modified fc
    from torchvision import models
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    try:
        model.load_state_dict(cleaned, strict=False)
        model = model.to(DEVICE)
        model.eval()
        print("✅ Model loaded with torchvision resnet50 fallback")
        return model
    except Exception as e:
        print(f"⚠  torchvision resnet50 fallback failed: {e}")

    raise RuntimeError("Could not load the model with any known architecture")

print("🔄 Loading model...")
model = load_model()
print("✅ Model ready!")


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
        # Read and preprocess image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        # Build results
        results = []
        for i, class_name in enumerate(CLASS_NAMES):
            results.append({
                "class": class_name,
                "confidence": round(probabilities[i].item() * 100, 2),
            })

        # Sort by confidence descending
        results.sort(key=lambda x: x["confidence"], reverse=True)

        return jsonify({
            "success": True,
            "prediction": results[0]["class"],
            "confidence": results[0]["confidence"],
            "all_predictions": results,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
