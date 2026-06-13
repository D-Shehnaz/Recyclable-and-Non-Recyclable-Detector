import cv2

def dip_preprocess(img):
    img = cv2.resize(img, (640, 640))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    equalized = cv2.equalizeHist(blur)

    return img  # YOLO uses original resized image