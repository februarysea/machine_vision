# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainWindowUI.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1013, 700)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btnpProcess = QtWidgets.QPushButton(self.centralwidget)
        self.btnpProcess.setGeometry(QtCore.QRect(870, 170, 112, 32))
        self.btnpProcess.setObjectName("btnpProcess")
        self.labelCamera = QtWidgets.QLabel(self.centralwidget)
        self.labelCamera.setGeometry(QtCore.QRect(40, 50, 800, 600))
        self.labelCamera.setAlignment(QtCore.Qt.AlignCenter)
        self.labelCamera.setObjectName("labelCamera")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(850, 30, 20, 641))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.btnpProcess.clicked.connect(MainWindow.btnProcess_clicked)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btnpProcess.setText(_translate("MainWindow", "处理"))
        self.labelCamera.setText(_translate("MainWindow", "摄像头图像"))


