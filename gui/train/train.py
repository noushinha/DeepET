# -*- coding: utf-8 -*-
# Form implementation generated from reading ui file 'train.ui'
# Created by: PyQt5 UI code generator 5.13.2
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(660, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 10, 631, 397))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.modelL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.modelL.setObjectName("modelL")
        self.gridLayout_2.addWidget(self.modelL, 3, 0, 1, 1)
        self.lossL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.lossL.setObjectName("lossL")
        self.gridLayout_2.addWidget(self.lossL, 7, 0, 1, 1)
        self.classnames = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.classnames.setObjectName("classnames")
        self.gridLayout_2.addWidget(self.classnames, 1, 1, 1, 1)
        # self.outputPath = QtWidgets.QLineEdit(self.gridLayoutWidget)
        # self.outputPath.setObjectName("outputPath")
        # self.gridLayout_2.addWidget(self.outputPath, 2, 1, 1, 1)
        self.basePathL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.basePathL.setObjectName("basePathL")
        self.gridLayout_2.addWidget(self.basePathL, 0, 0, 1, 1)
        self.LRL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.LRL.setObjectName("LRL")
        self.gridLayout_2.addWidget(self.LRL, 4, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.LR = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.LR.setObjectName("LR")
        self.horizontalLayout.addWidget(self.LR)
        self.epochL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.epochL.setObjectName("epochL")
        self.horizontalLayout.addWidget(self.epochL)
        self.epochs = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.epochs.setObjectName("epochs")
        self.horizontalLayout.addWidget(self.epochs)
        self.batchL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.batchL.setObjectName("batchL")
        self.horizontalLayout.addWidget(self.batchL)
        self.batchsize = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.batchsize.setObjectName("batchsize")
        self.horizontalLayout.addWidget(self.batchsize)
        self.patchL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.patchL.setObjectName("patchL")
        self.horizontalLayout.addWidget(self.patchL)
        self.patchsize = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.patchsize.setObjectName("patchsize")
        self.horizontalLayout.addWidget(self.patchsize)
        self.gridLayout_2.addLayout(self.horizontalLayout, 4, 1, 1, 1)
        self.basePath = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.basePath.setObjectName("basePath")
        self.gridLayout_2.addWidget(self.basePath, 0, 1, 1, 1)
        self.textEdit = QtWidgets.QTextEdit(self.gridLayoutWidget)
        self.textEdit.setMaximumSize(QtCore.QSize(16777215, 150))
        self.textEdit.setObjectName("textEdit")
        self.gridLayout_2.addWidget(self.textEdit, 9, 0, 1, 2)
        self.widthL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.widthL.setObjectName("widthL")
        self.gridLayout_2.addWidget(self.widthL, 6, 0, 1, 1)
        self.horizontalLayout9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout9.setObjectName("horizontalLayout9")
        self.width = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.width.setObjectName("Width")
        self.horizontalLayout9.addWidget(self.width)
        self.heightL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.heightL.setObjectName("heightL")
        self.horizontalLayout9.addWidget(self.heightL)
        self.height = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.height.setObjectName("Height")
        self.horizontalLayout9.addWidget(self.height)
        self.depthL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.depthL.setObjectName("depthL")
        self.horizontalLayout9.addWidget(self.depthL)
        self.depth = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.depth.setObjectName("depth")
        self.horizontalLayout9.addWidget(self.depth)
        self.gridLayout_2.addLayout(self.horizontalLayout9, 6, 1, 1, 1)
        self.trainBtn = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.trainBtn.setMaximumSize(QtCore.QSize(100, 16777215))
        self.trainBtn.setObjectName("trainBtn")
        self.gridLayout_2.addWidget(self.trainBtn, 8, 0, 1, 2)
        self.classNamesL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.classNamesL.setObjectName("classNamesL")
        self.gridLayout_2.addWidget(self.classNamesL, 1, 0, 1, 1)
        # self.outputPathL = QtWidgets.QLabel(self.gridLayoutWidget)
        # self.outputPathL.setObjectName("outputPathL")
        # self.gridLayout_2.addWidget(self.outputPathL, 2, 0, 1, 1)
        self.optimizerL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.optimizerL.setObjectName("optimizerL")
        self.gridLayout_2.addWidget(self.optimizerL, 5, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 660, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.modelL.setText(_translate("MainWindow", "model:"))
        self.lossL.setText(_translate("MainWindow", "loss:"))
        self.basePathL.setText(_translate("MainWindow", "root directory:"))
        self.LRL.setText(_translate("MainWindow", "learning rate:"))
        self.LR.setText(_translate("MainWindow", "0.001"))
        self.epochL.setText(_translate("MainWindow", "epochs:"))
        self.epochs.setText(_translate("MainWindow", "10"))
        self.batchL.setText(_translate("MainWindow", "batch size:"))
        self.batchsize.setText(_translate("MainWindow", "16"))
        self.patchL.setText(_translate("MainWindow", "patch size:"))
        self.patchsize.setText(_translate("MainWindow", "64"))
        self.trainBtn.setText(_translate("MainWindow", "train"))
        self.classNamesL.setText(_translate("MainWindow", "Class names:"))
        self.optimizerL.setText(_translate("MainWindow", "Optimizer:"))
        self.widthL.setText(_translate("MainWindow", "Width:"))
        self.heightL.setText(_translate("MainWindow", "Height:"))
        self.depthL.setText(_translate("MainWindow", "Depth:"))
        self.width.setText(_translate("MainWindow", "512"))
        self.height.setText(_translate("MainWindow", "512"))
        self.depth.setText(_translate("MainWindow", "200"))
        self.classnames.setText(_translate("MainWindow", "1bxn, 1qvr, 1s3x, 1u6g, 2cg9, 3cf3, 3d2f, 3gl1, 3h84, 3qm1, 4b4t, 4d8q"))
        self.basePath.setText(_translate("MainWindow", "/data2/"))
