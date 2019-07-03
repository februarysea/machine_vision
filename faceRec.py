import cv2
import os
import numpy as np


def LoadImages(data):
    '''
    加载图片数据用于训练
    params:
        data:训练数据所在的目录，要求图片尺寸一样
    ret:
        images:[m,height,width]  m为样本数，height为高，width为宽
        names：名字的集合
        labels：标签
    '''
    images = []
    names = []
    labels = []

    label = 0

    # 遍历所有文件夹
    for subdir in os.listdir(data):
        # os.path.join 路径拼接
        subpath = os.path.join(data, subdir)
        # 判断是否是目录
        if os.path.isdir(subpath):
            # 在每一个文件夹中存放着一个人的许多照片
            names.append(subdir)
            # 遍历文件夹中的图片文件
            for filename in os.listdir(subpath):
                if filename.startswith("."):
                    continue
                imgpath = os.path.join(subpath, filename)
                img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append(gray_img)
                labels.append(label)
                print("label:" + str(label) + " img:" + str(filename))
            label += 1
    images = np.asarray(images)
    labels = np.asarray(labels)
    return images, labels, names


# 检验训练结果
def FaceRec(data):
    # 加载训练的数据
    X, y, names = LoadImages(data)
    # print('x',X)
    model = cv2.face.EigenFaceRecognizer_create()
    model.train(X, y)

    # 打开摄像头
    camera = cv2.VideoCapture(0)
    cv2.namedWindow('Dynamic')

    # 创建级联分类器
    face_casecade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True:
        # 读取一帧图像
        # ret:图像是否读取成功
        # frame：该帧图像
        ret, frame = camera.read()
        # 判断图像是否读取成功
        # print('ret',ret)
        if ret:
            # 转换为灰度图
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 利用级联分类器鉴别人脸
            faces = face_casecade.detectMultiScale(gray_img, 1.3, 5)

            # 遍历每一帧图像，画出矩形
            for (x, y, w, h) in faces:
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 蓝色
                roi_gray = gray_img[y:y + h, x:x + w]

                try:
                    # 将图像转换为宽64 高64的图像
                    # resize（原图像，目标大小，（插值方法）interpolation=，）
                    roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_LINEAR)
                    params = model.predict(roi_gray)
                    print('Label:%s,confidence:%.2f' % (params[0], params[1]))
                    '''
                    putText:给照片添加文字
                    putText(输入图像，'所需添加的文字'，左上角的坐标，字体，字体大小，颜色，字体粗细)
                    '''
                    if params[1] < 2000:
                        cv2.putText(frame, "none", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                    else:
                        cv2.putText(frame, names[params[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

                except:
                    continue

            cv2.imshow('Dynamic', frame)

            # 按下esc键退出
            if cv2.waitKey(100) & 0xFF == 27:
                break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    data = "training-data"
    FaceRec(data)