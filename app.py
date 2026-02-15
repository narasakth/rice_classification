import os
import numpy as np
import onnxruntime as ort
from PIL import Image
from flask import Flask, request, jsonify, render_template

# ── App Configuration ──────────────────────────────────────────
app = Flask(__name__)

# ใช้ path ให้ตรงกับโครงสร้างไฟล์ของคุณ
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "rice_model.onnx")

# ชื่อคลาสต้องเรียงลำดับเหมือนตอนเทรนเป๊ะๆ
CLASS_NAMES = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

# ── Load ONNX Model (โหลดครั้งเดียวตอนเริ่มแอป) ────────────────
print("🔄 Loading ONNX model...")
# ใช้ onnxruntime แทน pytorch (กิน RAM น้อยกว่ามาก)
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
print("✅ Model loaded successfully")

# ── Helper Functions ───────────────────────────────────────────

def softmax(x):
    """คำนวณ Softmax เพื่อหาค่าความมั่นใจ (%) โดยใช้ Numpy"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def preprocess_image(image):
    """แปลงรูปภาพให้เป็น format ที่โมเดลต้องการ (โดยไม่ใช้ torchvision)"""
    # 1. Resize เป็น 224x224
    image = image.resize((224, 224))
    
    # 2. แปลงเป็น Numpy Array และทำให้ค่าเป็น 0-1 (float32)
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # 3. Normalize (ค่ามาตรฐานของ ImageNet)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # 4. Transpose จาก (H, W, C) -> (C, H, W)
    img_array = img_array.transpose(2, 0, 1)
    
    # 5. เพิ่มมิติ Batch: (1, 3, 224, 224)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

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
        # เปิดรูปภาพ
        img = Image.open(file.stream).convert("RGB")
        
        # เตรียมรูปภาพ (Preprocess)
        input_tensor = preprocess_image(img)

        # รันโมเดล (Inference)
        outputs = session.run(None, {input_name: input_tensor})
        raw_scores = outputs[0][0]  # ดึงค่าผลลัพธ์ออกมา

        # คำนวณความน่าจะเป็น
        probs = softmax(raw_scores)

        # จัดเตรียมผลลัพธ์ส่งกลับ
        results = []
        for i, score in enumerate(probs):
            results.append({
                "class": CLASS_NAMES[i],
                "confidence": round(float(score) * 100, 2)
            })
        
        # เรียงลำดับจากมากไปน้อย
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
    # Render จะส่งค่า PORT มาให้ทาง Environment Variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)