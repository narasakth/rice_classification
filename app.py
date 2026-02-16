import streamlit as st
import os
import numpy as np
import onnxruntime as ort
from PIL import Image

# ── 1. Page Configuration ─────────────────────────────────────
st.set_page_config(page_title="Rice Classification", page_icon="🌾")
st.title("🌾 Rice Variety Classification")
st.markdown("อัปโหลดรูปเมล็ดข้าวเพื่อให้ AI ช่วยจำแนกสายพันธุ์")

# ── 2. Load Model (Cached) ────────────────────────────────────
# ใช้ st.cache_resource เพื่อให้โหลดโมเดลแค่ครั้งเดียว ไม่โหลดใหม่ทุกครั้งที่กดปุ่ม
@st.cache_resource
def load_session():
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "rice_model.onnx")
    return ort.InferenceSession(MODEL_PATH)

try:
    session = load_session()
    input_name = session.get_inputs()[0].name
    CLASS_NAMES = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
except Exception as e:
    st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")

# ── 3. Helper Functions ───────────────────────────────────────
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def preprocess_image(image):
    # 1. Resize เป็น 224x224
    image = image.resize((224, 224))
    # 2. แปลงเป็น Numpy Array (0-1)
    img_array = np.array(image).astype(np.float32) / 255.0
    # 3. Normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    # 4. Transpose (H,W,C -> C,H,W) และเพิ่ม Batch Dim
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ── 4. User Interface ─────────────────────────────────────────
uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพเมล็ดข้าว...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # แสดงรูปที่ผู้ใช้อัปโหลด
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="รูปภาพที่อัปโหลด", use_container_width=True)

    with col2:
        with st.spinner('กำลังประมวลผล...'):
            # Preprocess & Predict
            input_tensor = preprocess_image(image)
            outputs = session.run(None, {input_name: input_tensor})
            probs = softmax(outputs[0][0])

            # สรุปผลลัพธ์
            results = []
            for i, score in enumerate(probs):
                results.append({
                    "class": CLASS_NAMES[i],
                    "confidence": float(score)
                })
            results.sort(key=lambda x: x["confidence"], reverse=True)

            # แสดงผลลัพธ์ตัวที่มั่นใจที่สุด
            top_result = results[0]
            st.success(f"ผลการพยากรณ์: **{top_result['class']}**")
            st.metric("ความมั่นใจ", f"{top_result['confidence']*100:.2f}%")

    # แสดงกราฟแท่งเปรียบเทียบทุกคลาส
    st.divider()
    st.subheader("รายละเอียดความมั่นใจทั้งหมด")
    chart_data = {res["class"]: res["confidence"] for res in results}
    st.bar_chart(chart_data)

else:
    st.info("💡 กรุณาอัปโหลดรูปภาพเพื่อเริ่มการวิเคราะห์")