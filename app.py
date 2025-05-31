import streamlit as st
import torch
import os
import sys
import cv2
import numpy as np
from PIL import Image

# Add yolov5 path to system path
YOLOV5_PATH = os.path.join(os.path.dirname(__file__), 'yolov5')
sys.path.append(YOLOV5_PATH)

# Import from yolov5
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# Load YOLOv5 model
@st.cache_resource
def load_model(weights_path='best.pt'):
    device = select_device('')
    model = attempt_load(weights_path, map_location=device)
    return model, device

# Inference function
def detect_dents(image, model, device):
    img = np.array(image)
    img0 = img.copy()
    img = cv2.resize(img, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # Normalize to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                label = f"Dent {conf:.2f}"
                xyxy = list(map(int, xyxy))
                cv2.rectangle(img0, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                cv2.putText(img0, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img0

# Streamlit UI
st.title("Car Dent Detection with YOLOv5")

uploaded_file = st.file_uploader("Upload an image of a car", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model, device = load_model()

    with st.spinner("Detecting dents..."):
        result_img = detect_dents(image, model, device)
        st.image(result_img, caption="Detection Result", use_column_width=True)
