import streamlit as st
from PIL import Image
import torch
import numpy as np
import os
import sys
# import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

from models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.torch_utils import select_device

device = select_device('cpu')
MODEL_PATH = 'weights/best.pt'
model = attempt_load(MODEL_PATH, map_location=device)
model.eval()

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

    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in det:
                label = f"Dent {conf:.2f}"
                c1, c2 = tuple(map(int, xyxy[:2])), tuple(map(int, xyxy[2:]))
                img = cv2.rectangle(img.copy(), c1, c2, (0, 255, 0), 2)
                img = cv2.putText(img, label, c1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    st.image(img, caption="Detected Dents", use_container_width=True)
