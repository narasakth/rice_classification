import torch
import timm
from PIL import Image
import numpy as np

# 1. โหลดโมเดลเดิมของคุณ (ตัวอย่างสมมติว่าเป็น ResNet50)
# ต้องแก้ชื่อรุ่น (model_name) และ num_classes ให้ตรงกับที่คุณเทรนมา
# เปลี่ยนชื่อโมเดลเป็น resnetv2_50
model = timm.create_model('resnetv2_50', pretrained=False, num_classes=5)
# ใส่ตัว r ไว้หน้าเครื่องหมายคำพูด '...'
model.load_state_dict(torch.load(r'C:\Consigliere\cnn\lab6\model\best_model_resnetv2.pth', map_location='cpu'))
model.eval()

# 2. สร้างข้อมูลจำลอง (Dummy Input) ขนาดเท่ารูปที่ใช้เทรน (เช่น 224x224)
dummy_input = torch.randn(1, 3, 224, 224)

# 3. สั่ง Export เป็น .onnx
save_path = "model/rice_model.onnx"
torch.onnx.export(model, 
                  dummy_input, 
                  save_path, 
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'])

print(f"✅ แปลงโมเดลเป็น {save_path} เรียบร้อยแล้ว!")