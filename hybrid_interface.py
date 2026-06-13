import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
from model_cnn import get_model

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Waste Detection System",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Hybrid YOLO + CNN Waste Detection")
st.write("Upload an image to detect and classify waste.")

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():

    # Use custom YOLO model if available
    # yolo = YOLO("yolo_model/best.pt")

    yolo = YOLO("yolov8n.pt")

    cnn = get_model().to(device)

    cnn.load_state_dict(
        torch.load(
            "cnn_model/model.pth",
            map_location=device
        )
    )

    cnn.eval()

    return yolo, cnn

yolo, cnn = load_models()

# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Waste Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    with st.spinner("Detecting waste..."):

        results = yolo(frame)

        detected_count = 0

        for r in results:

            for box in r.boxes:

                detected_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                h, w = frame.shape[:2]

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                crop_rgb = cv2.cvtColor(
                    crop,
                    cv2.COLOR_BGR2RGB
                )

                img = Image.fromarray(crop_rgb)
                img = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = cnn(img)
                    pred = torch.argmax(
                        output,
                        dim=1
                    ).item()

                if pred == 0:
                    label = "Recyclable"
                    color = (0, 255, 0)
                else:
                    label = "Non-Recyclable"
                    color = (0, 0, 255)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        result_image = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

    with col2:
        st.subheader("Detection Result")
        st.image(result_image, use_container_width=True)

    st.success(
        f"Processing completed. Objects detected: {detected_count}"
    )

    result_pil = Image.fromarray(result_image)

    st.download_button(
        label="📥 Download Result",
        data=uploaded_file.getvalue(),
        file_name="result.jpg",
        mime="image/jpeg"
    )