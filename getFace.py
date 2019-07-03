import cv2
import os
import random

out_dir = 'jch'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

'''
# 改变亮度与对比度
def relight(img, alpha=1, bias=0):
    w = img.shape[1]
    h = img.shape[0]
    for i in range(0, w):
        for j in range(0, h):
            for c in range(3):
                tmp = int(img[j, i, c]*alpha + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j, i, c] = tmp
    return img
'''


# 获取分类器
haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 打开摄像头 参数为输入流，可以为摄像头或视频文件
camera = cv2.VideoCapture(0)

n = 1
while 1:
    if n <= 100:
        print('It`s processing %s image.' % n)
        # 读帧
        success, img = camera.read()

        # 将图像转化为灰度图像
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # scaleFactor 图像尺寸减小的比例
        # minNeighobrs 目标监测到5次才算真正的目标
        faces = haar.detectMultiScale(image=gray_img, scaleFactor=1.3, minNeighbors=5)
        for f_x, f_y, f_w, f_h in faces:
            # 裁剪 剩下面部
            face = img[f_y:f_y+f_h, f_x:f_x+f_w]
            face = cv2.resize(src=face, dsize=(64, 64))
            cv2.imshow('img', face)
            # png rather than jpg
            cv2.imwrite(out_dir+'/'+str(n)+'.png', face)
            n += 1
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break

    else:
        break

