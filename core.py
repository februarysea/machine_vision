import cv2
import numpy as np
from matplotlib import pyplot as plt

# 图片人脸识别
def discern(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cap = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faceRects = cap.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))
    if len(faceRects):
        for faceRect in faceRects:
            x, y, w, h = faceRect
            cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)  # 框出人脸
    # cv2.imshow("Image", img)
    return img
