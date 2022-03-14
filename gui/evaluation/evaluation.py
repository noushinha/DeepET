# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'evaluation.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(647, 342)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 20, 601, 31))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.input_pathL = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.input_pathL.setObjectName("input_pathL")
        self.horizontalLayout.addWidget(self.input_pathL)
        self.input_path = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.input_path.setObjectName("input_path")
        self.horizontalLayout.addWidget(self.input_path)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(9, 60, 601, 31))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.model_pathL = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.model_pathL.setObjectName("model_pathL")
        self.horizontalLayout_2.addWidget(self.model_pathL)
        self.model_path = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.model_path.setObjectName("model_path")
        self.horizontalLayout_2.addWidget(self.model_path)
        self.horizontalLayoutWidget_4 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_4.setGeometry(QtCore.QRect(10, 100, 601, 31))
        self.horizontalLayoutWidget_4.setObjectName("horizontalLayoutWidget_4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_4)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.output_pathL = QtWidgets.QLabel(self.horizontalLayoutWidget_4)
        self.output_pathL.setObjectName("output_pathL")
        self.horizontalLayout_4.addWidget(self.output_pathL)
        self.output_path = QtWidgets.QLineEdit(self.horizontalLayoutWidget_4)
        self.output_path.setObjectName("output_path")
        self.horizontalLayout_4.addWidget(self.output_path)
        self.segtBtn = QtWidgets.QPushButton(self.centralwidget)
        self.segtBtn.setGeometry(QtCore.QRect(95, 230, 80, 25))
        self.segtBtn.setObjectName("segtBtn")
        self.clusBtn = QtWidgets.QPushButton(self.centralwidget)
        self.clusBtn.setGeometry(QtCore.QRect(310, 230, 80, 25))
        self.clusBtn.setObjectName("clusBtn")
        self.evalBtn = QtWidgets.QPushButton(self.centralwidget)
        self.evalBtn.setGeometry(QtCore.QRect(525, 230, 80, 25))
        self.evalBtn.setObjectName("evalBtn")
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(10, 130, 601, 51))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.modelL = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        # self.modelL.setMaximumSize(QtCore.QSize(200, 16777215))
        self.modelL.setObjectName("modelL")
        self.horizontalLayout_3.addWidget(self.modelL)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.input_pathL.setText(_translate("MainWindow", "Input  Path"))
        self.input_path.setText(_translate("MainWindow", "/data2/results/RealData/evaluation/23_resampled.mrc"))
        # self.input_path.setText(_translate("MainWindow", "/data2/results/RealData/evaluation/8_resampled.mrc"))
        # self.input_path.setText(_translate("MainWindow", "/data2/results/SyntheticData/evaluation/target_grandmodel_9.mrc"))
        self.model_pathL.setText(_translate("MainWindow", "Model Path"))
        # self.model_path.setText(_translate("MainWindow", "/data2/results/RealData/evaluation/model_final_weights_PT_No3.h5"))
        self.model_path.setText(_translate("MainWindow", "/data2/results/RealData/evaluation/model_final_weights_RB_No3.h5"))
        self.output_pathL.setText(_translate("MainWindow", "Output Path"))
        self.output_path.setText(_translate("MainWindow", "/data2/results/RealData/evaluation/"))
        # self.output_path.setText(_translate("MainWindow", "/data2/results/SyntheticData/evaluation/"))
        self.segtBtn.setText(_translate("MainWindow", "segment"))
        self.clusBtn.setText(_translate("MainWindow", "cluster"))
        self.evalBtn.setText(_translate("MainWindow", "evaluate"))
        self.modelL.setText(_translate("MainWindow", "CNN Model"))
        # self.UnetrBtn.setText(_translate("MainWindow", "3D Unet"))
        # self.RPNrBtn.setText(_translate("MainWindow", "RPN"))
