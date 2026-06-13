# ♻️ Recyclable and Non-Recyclable Waste Detection System

A Hybrid Deep Learning-based Waste Detection System that combines **YOLOv8 Object Detection** with **CNN Classification** to identify waste objects and classify them as **Recyclable** or **Non-Recyclable**.

The system also provides a user-friendly **Streamlit Web Application** for real-time image upload, waste detection, classification, and result visualization.

---

## 📌 Features

* Object Detection using YOLOv8
* Waste Classification using CNN
* Hybrid YOLO + CNN Pipeline
* Image Preprocessing using Digital Image Processing (DIP)
* Streamlit Web Interface
* Bounding Box Visualization
* Download Detection Results
* GPU Support (CUDA)

---

## 🏗️ Project Architecture

```text
Input Image
      │
      ▼
Image Preprocessing (DIP)
      │
      ▼
YOLOv8 Object Detection
      │
      ▼
Detected Object Cropping
      │
      ▼
CNN Classification
      │
      ├── Recyclable
      └── Non-Recyclable
      │
      ▼
Result Visualization
```

---

## 📂 Project Structure

```text
Recyclable-and-Non-Recyclable-Detector/
│
├── app.py
├── model_cnn.py
├── train_cnn.py
├── train_yolo.py
├── dip_preprocessing.py
│
├── cnn_model/
│   └── model.pth
│
├── yolo_model/
│   └── best.pt
│
├── dataset_cnn/
│   └── images/
│       └── train/
│           ├── recyclable/
│           └── non_recyclable/
│
├── dataset.yaml
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

* Python
* PyTorch
* YOLOv8 (Ultralytics)
* OpenCV
* Streamlit
* NumPy
* Pillow
* TorchVision

---

## 📦 Installation

### Clone Repository

```bash
git clone https://github.com/D-Shehnaz/Recyclable-and-Non-Recyclable-Detector.git

cd Recyclable-and-Non-Recyclable-Detector
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 CNN Training

The CNN model is trained on categorized waste images.

Dataset Structure:

```text
dataset_cnn/
└── images/
    └── train/
        ├── recyclable/
        └── non_recyclable/
```

Run training:

```bash
python train_cnn.py
```

Output:

```text
cnn_model/model.pth
```

---

## 🎯 YOLOv8 Training

Train YOLOv8 on waste object detection dataset.

```bash
python train_yolo.py
```

or

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
```

Output:

```text
runs/detect/train/
```

Best model:

```text
runs/detect/train/weights/best.pt
```

---

## 🖼️ Digital Image Processing

Before detection, images are preprocessed using:

* Image Resizing
* Grayscale Conversion
* Gaussian Blur
* Histogram Equalization

Example:

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
equalized = cv2.equalizeHist(blur)
```

---

## 🚀 Running the Application

Start Streamlit:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## 📸 Application Workflow

1. Upload Waste Image
2. YOLO Detects Waste Objects
3. Objects are Cropped
4. CNN Classifies Each Object
5. Bounding Boxes are Drawn
6. Result is Displayed
7. User Can Download Output Image

---

## 📊 Classes

### CNN Classes

| Label | Description    |
| ----- | -------------- |
| 0     | Recyclable     |
| 1     | Non-Recyclable |

---

## 💻 Hardware Requirements

Recommended:

* NVIDIA GPU
* CUDA Supported Device
* 8GB+ RAM

Minimum:

* CPU
* 4GB RAM

---

## 🔮 Future Improvements

* Real-time Webcam Detection
* Video Waste Detection
* Multi-Class Waste Classification
* Mobile Application Deployment
* Cloud Deployment
* Improved Detection Accuracy

---

## 👩‍💻 Author

**Shehnaz Begum**

GitHub:
https://github.com/D-Shehnaz

---

## 📜 License

This project is developed for educational and research purposes.

Feel free to use and modify the project with proper attribution.
