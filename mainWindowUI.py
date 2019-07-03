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
        MainWindow.resize(1000, 700)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.labelCamera = QtWidgets.QLabel(self.centralwidget)
        self.labelCamera.setGeometry(QtCore.QRect(40, 50, 800, 600))
        self.labelCamera.setAlignment(QtCore.Qt.AlignCenter)
        self.labelCamera.setObjectName("labelCamera")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(850, 30, 20, 641))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(887, 55, 91, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.labeldata = QtWidgets.QLabel(self.centralwidget)
        self.labeldata.setGeometry(QtCore.QRect(890, 120, 71, 321))
        self.labeldata.setObjectName("labeldata")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.labelCamera.setText(_translate("MainWindow", "摄像头图像"))
        self.label.setText(_translate("MainWindow", "识别结果"))
        self.labeldata.setText(_translate("MainWindow", "等待识别....."))


