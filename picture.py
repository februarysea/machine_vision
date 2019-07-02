# -*- coding:utf-8 -*-
# OpenCV版本的视频检测
import cv2

# 获取摄像头0表示第一个摄像头
cap = cv2.VideoCapture(0)
while (1):  # 逐帧显示
    ret, img = cap.read()
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()  # 释放摄像头
cv2.destroyAllWindows()  # 释放窗口资源
