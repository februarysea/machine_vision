import sys
import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QMainWindow
from mainWindowUI import Ui_MainWindow
import faceRec


class PyqtWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.camera = cv2.VideoCapture(0)
        self.is_camera_opened = False  # 摄像头有没有打开标记
        self.is_face_opened = False  # 动态人脸识别是否打开

        # 定时器：16ms捕获一帧
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._queryFrame)
        self._timer.setInterval(16)

        self._timer.start()

    @QtCore.pyqtSlot()
    def _queryFrame(self):
        #  循环捕获图片
        ret, self.frame = self.camera.read()
        model, names = faceRec.FaceRec(data="training-data")
        self.frame = faceRec.show(model=model, names=names)
        # rows: Y cols: X
        img_rows, img_cols, channels = self.frame.shape
        # 每行byte数
        bytesPerLine = channels * img_cols

        # BGR -> RGB
        cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB, self.frame)
        QImg = QImage(self.frame.data, img_cols, img_rows, bytesPerLine, QImage.Format_RGB888)
        self.labelCamera.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelCamera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnProcess_clicked(self):
        pass

    def closeEvent(self, QCloseEvent):
        self._timer.stop()
        self.camera.release()  # 释放摄像头
        cv2.destroyAllWindows()  # 释放窗口资源


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = PyqtWindow()
    window.show()
    sys.exit(app.exec_())
