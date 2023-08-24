# -*- coding: utf-8 -*-
# Form implementation generated from reading ui file 'train.ui'
# Created by: PyQt5 UI code generator 5.13.2
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(830, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 10, 791, 397))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.trainL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.trainL.setObjectName("trainL")
        self.gridLayout_2.addWidget(self.trainL, 8, 0, 1, 1)
        self.modelL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.modelL.setObjectName("modelL")
        self.gridLayout_2.addWidget(self.modelL, 9, 0, 1, 1)
        self.lossL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.lossL.setObjectName("lossL")
        self.gridLayout_2.addWidget(self.lossL, 11, 0, 1, 1)
        self.LRType = QtWidgets.QLabel(self.gridLayoutWidget)
        self.LRType.setObjectName("LRType")
        self.gridLayout_2.addWidget(self.LRType, 12, 0, 1, 1)
        self.AugType = QtWidgets.QLabel(self.gridLayoutWidget)
        self.AugType.setObjectName("AugType")
        self.gridLayout_2.addWidget(self.AugType, 13, 0, 1, 1)
        self.classnames = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.classnames.setObjectName("classnames")
        self.gridLayout_2.addWidget(self.classnames, 3, 1, 1, 1)
        self.outputPath = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.outputPath.setObjectName("outputPath")
        self.gridLayout_2.addWidget(self.outputPath, 1, 1, 1, 1)
        self.basePathL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.basePathL.setObjectName("basePathL")
        self.gridLayout_2.addWidget(self.basePathL, 0, 0, 1, 1)
        self.weightPathL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.weightPathL.setObjectName("weightPathL")
        self.gridLayout_2.addWidget(self.weightPathL, 2, 0, 1, 1)
        self.LRL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.LRL.setObjectName("LRL")
        self.gridLayout_2.addWidget(self.LRL, 5, 0, 1, 1)
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
        self.gridLayout_2.addLayout(self.horizontalLayout, 5, 1, 1, 1)
        self.basePath = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.basePath.setObjectName("basePath")
        self.gridLayout_2.addWidget(self.basePath, 0, 1, 1, 1)
        self.textEdit = QtWidgets.QTextEdit(self.gridLayoutWidget)
        self.textEdit.setMaximumSize(QtCore.QSize(16777215, 150))
        self.textEdit.setObjectName("textEdit")
        self.weightPath = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.weightPath.setObjectName("weightPath")
        self.gridLayout_2.addWidget(self.weightPath, 2, 1, 1, 1)
        self.gridLayout_2.addWidget(self.textEdit, 20, 0, 1, 2)
        self.widthL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.widthL.setObjectName("widthL")
        self.gridLayout_2.addWidget(self.widthL, 4, 0, 1, 1)
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
        self.gridLayout_2.addLayout(self.horizontalLayout9, 4, 1, 1, 1)



        self.AugmentL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.AugmentL.setObjectName("AugmentL")
        self.gridLayout_2.addWidget(self.AugmentL, 7, 0, 1, 1)
        self.horizontalLayout10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout10.setObjectName("horizontalLayout10")
        self.augment = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.augment.setObjectName("augment")
        self.horizontalLayout10.addWidget(self.augment)
        self.ValidationL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.ValidationL.setObjectName("ValidationL")
        self.horizontalLayout10.addWidget(self.ValidationL)
        self.validation = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.validation.setObjectName("validation")
        self.horizontalLayout10.addWidget(self.validation)
        self.gridLayout_2.addLayout(self.horizontalLayout10, 7, 1, 1, 1)



        self.trainBtn = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.trainBtn.setMaximumSize(QtCore.QSize(100, 16777215))
        self.trainBtn.setObjectName("trainBtn")
        self.gridLayout_2.addWidget(self.trainBtn, 15, 1, 2, 2)
        self.classNamesL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.classNamesL.setObjectName("classNamesL")
        self.gridLayout_2.addWidget(self.classNamesL, 3, 0, 1, 1)
        self.outputPathL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.outputPathL.setObjectName("outputPathL")
        self.gridLayout_2.addWidget(self.outputPathL, 1, 0, 1, 1)
        self.optimizerL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.optimizerL.setObjectName("optimizerL")
        self.gridLayout_2.addWidget(self.optimizerL, 10, 0, 1, 1)
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
        self.trainL.setText(_translate("MainWindow", "training type:"))
        self.modelL.setText(_translate("MainWindow", "model:"))
        self.lossL.setText(_translate("MainWindow", "loss:"))
        self.LRType.setText(_translate("MainWindow", "LR type:"))
        self.AugType.setText(_translate("MainWindow", "Augmentation:"))
        self.basePathL.setText(_translate("MainWindow", "root directory:"))
        self.outputPathL.setText(_translate("MainWindow", "Output directory:"))
        self.weightPathL.setText(_translate("MainWindow", "model weight directory:"))
        self.LRL.setText(_translate("MainWindow", "learning rate:"))
        self.LR.setText(_translate("MainWindow", "0.0001"))
        self.epochL.setText(_translate("MainWindow", "epochs:"))
        self.epochs.setText(_translate("MainWindow", "50"))
        self.batchL.setText(_translate("MainWindow", "batch size:"))
        self.batchsize.setText(_translate("MainWindow", "12"))
        self.patchL.setText(_translate("MainWindow", "patch size:"))
        self.patchsize.setText(_translate("MainWindow", "32"))
        self.trainBtn.setText(_translate("MainWindow", "train"))
        self.classNamesL.setText(_translate("MainWindow", "Class names:"))
        self.optimizerL.setText(_translate("MainWindow", "Optimizer:"))
        self.widthL.setText(_translate("MainWindow", "Width:"))
        self.heightL.setText(_translate("MainWindow", "Height:"))
        self.depthL.setText(_translate("MainWindow", "Depth:"))
        self.width.setText(_translate("MainWindow", "1000"))  # R 409, V2 451, V3 409
        self.height.setText(_translate("MainWindow", "1000"))  # R 409, V2 451, V3 409
        self.depth.setText(_translate("MainWindow", "250"))  # R 154, V2 146, V3 102
        self.AugmentL.setText(_translate("MainWindow", "Augmentation (%):"))
        self.ValidationL.setText(_translate("MainWindow", "Validation (%):"))
        self.augment.setText(_translate("MainWindow", "0.0"))
        self.validation.setText(_translate("MainWindow", ".1"))
        # self.classnames.setText(_translate("MainWindow", "1bxn, 1qvr, 1s3x, 1u6g, 2cg9, 3cf3, 3d2f, 3gl1, 3h84, 3qm1, 4b4t, 4d8q"))
        # self.classnames.setText(_translate("MainWindow", "4d8q"))
        # self.classnames.setText(_translate("MainWindow", "1bxn"))
        # self.classnames.setText(_translate("MainWindow", "3gl1"))
        # self.classnames.setText(_translate("MainWindow", "4d8q, 1bxn, 3gl1"))
        self.classnames.setText(_translate("MainWindow", "pt, rb"))  # invitro
        self.basePath.setText(_translate("MainWindow", "/data2/"))
        # self.outputPath.setText(_translate("MainWindow", "/data2/results/RealData/"))
        self.outputPath.setText(_translate("MainWindow", "/data2/results/SyntheticData/"))
        # self.weightPath.setText(_translate("MainWindow", "/data2/results/RealData/SOnR150EV3BottleNeckFreezedCyclicShapeMask/model_final_weights.h5"))
        self.weightPath.setText(_translate("MainWindow", ""))
        # self.weightPath.setText(_translate("MainWindow", "/data2/results/SyntheticData/FewShotLearningV3R/model_final_weights.h5"))
        # self.weightPath.setText(_translate("MainWindow", "/data2/results/SyntheticData/FreshTrainOnV3/model_final_weights.h5"))
        # self.weightPath.setText(_translate("MainWindow", "/data2/results/SyntheticData/FreshTrainOnV3ShapedMask/model_final_weights.h5"))
        # self.weightPath.setText(_translate("MainWindow", "/data2/results/SyntheticData/FreshTrainOnV2HighDoseLowDoseShapedMask/model_final_weights.h5"))
        self.textEdit.setText(_translate("MainWindow", "Training iterations..."))
        # self.weightPath.setText(_translate("MainWindow", "/data2/results/RealData/evaluation/model_final_weights_PTRB_No3.h5"))
        # self.weightPath.setText(_translate("MainWindow", "/data2/results/RealData/evaluation/model_final_weights_TL_decoder.h5"))

