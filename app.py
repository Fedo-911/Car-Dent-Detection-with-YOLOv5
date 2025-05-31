import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import os
import sys

# Add yolov5 to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

from models.experimental import attempt_load
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))
from yolov5.utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.torch_utils import select_device

# Device and model setup
device = select_device('cpu')
MODEL_PATH = 'weights/best.pt'
model = attempt_load(MODEL_PATH, map_location=device)
model.eval()

# Streamlit UI
st.title("🚗 Car Dent Detection with YOLOv5")
st.write("Upload a car image to detect dents using your trained model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    img_resized = letterbox(img, new_shape=640)[0]
    img_resized = img_resized.transpose((2, 0, 1))[::-1]
    img_resized = np.ascontiguousarray(img_resized)

    img_tensor = torch.from_numpy(img_resized).to(device)
    img_tensor = img_tensor.float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    with st.spinner("Detecting dents..."):
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    img_pil = image.copy()
    draw = ImageDraw.Draw(img_pil)

    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in det:
                label = f"Dent {conf:.2f}"
                x1, y1, x2, y2 = map(int, xyxy)
                draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
                draw.text((x1, y1 - 10), label, fill="green")

    st.image(img_pil, caption="Detected Dents", use_container_width=True)
