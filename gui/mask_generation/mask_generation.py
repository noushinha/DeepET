# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mask_generation.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1541, 1298)
        MainWindow.setMinimumSize(QtCore.QSize(16777215, 16777215))
        MainWindow.setMaximumSize(QtCore.QSize(16777215, 16777215))
        MainWindow.setMouseTracking(False)
        MainWindow.setAutoFillBackground(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 10, 1500, 1200))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.gridLayout.setContentsMargins(9, 9, 9, 9)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.annTable = QtWidgets.QTableWidget(self.gridLayoutWidget)
        self.annTable.setMinimumSize(QtCore.QSize(360, 512))
        self.annTable.setMaximumSize(QtCore.QSize(460, 16777215))
        self.annTable.setRowCount(50)
        self.annTable.setColumnCount(4)
        self.annTable.setObjectName("annTable")
        item = QtWidgets.QTableWidgetItem()
        self.annTable.setHorizontalHeaderItem(0, item)
        self.gridLayout.addWidget(self.annTable, 5, 0, 1, 1)
        self.outputPathL = QtWidgets.QLabel(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.outputPathL.sizePolicy().hasHeightForWidth())
        self.outputPathL.setSizePolicy(sizePolicy)
        self.outputPathL.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.outputPathL.setObjectName("outputPathL")
        self.gridLayout.addWidget(self.outputPathL, 4, 0, 1, 1)
        self.radiLenL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.radiLenL.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.radiLenL.setObjectName("radiLenL")
        self.gridLayout.addWidget(self.radiLenL, 2, 0, 1, 1)
        self.radiLen = QtWidgets.QLineEdit(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.radiLen.sizePolicy().hasHeightForWidth())
        self.radiLen.setSizePolicy(sizePolicy)
        self.radiLen.setMaximumSize(QtCore.QSize(450, 16777215))
        self.radiLen.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.radiLen.setObjectName("radiLen")
        self.gridLayout.addWidget(self.radiLen, 2, 1, 1, 1)
        self.showAnnL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.showAnnL.setObjectName("showAnnL")
        self.gridLayout.addWidget(self.showAnnL, 3, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.gridLayoutWidget)
        self.groupBox.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.CircRBtn = QtWidgets.QRadioButton(self.groupBox)
        self.CircRBtn.setGeometry(QtCore.QRect(0, 0, 100, 31))
        self.CircRBtn.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.CircRBtn.setChecked(True)
        self.CircRBtn.setObjectName("CircRBtn")
        self.RecRBtn = QtWidgets.QRadioButton(self.groupBox)
        self.RecRBtn.setGeometry(QtCore.QRect(170, 0, 161, 31))
        self.RecRBtn.setObjectName("RecRBtn")
        self.gridLayout.addWidget(self.groupBox, 1, 1, 1, 1)
        self.inputPathHLayout = QtWidgets.QHBoxLayout()
        self.inputPathHLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.inputPathHLayout.setContentsMargins(0, -1, -1, -1)
        self.inputPathHLayout.setSpacing(6)
        self.inputPathHLayout.setObjectName("inputPathHLayout")
        self.inputPath = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.inputPath.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.inputPath.sizePolicy().hasHeightForWidth())
        self.inputPath.setSizePolicy(sizePolicy)
        self.inputPath.setMaximumSize(QtCore.QSize(460, 16777215))
        self.inputPath.setCursorPosition(20)
        self.inputPath.setDragEnabled(True)
        self.inputPath.setObjectName("inputPath")
        self.inputPathHLayout.addWidget(self.inputPath)
        self.loadBtn = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.loadBtn.setMaximumSize(QtCore.QSize(120, 16777215))
        self.loadBtn.setObjectName("loadBtn")
        self.inputPathHLayout.addWidget(self.loadBtn)
        self.inputPathEmptyL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.inputPathEmptyL.setText("")
        self.inputPathEmptyL.setObjectName("inputPathEmptyL")
        self.inputPathHLayout.addWidget(self.inputPathEmptyL)
        self.gridLayout.addLayout(self.inputPathHLayout, 0, 1, 1, 1)
        self.ShowAnnBtn = QtWidgets.QCheckBox(self.gridLayoutWidget)
        self.ShowAnnBtn.setObjectName("ShowAnnBtn")
        self.gridLayout.addWidget(self.ShowAnnBtn, 3, 1, 1, 1)
        self.inputPathL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.inputPathL.setObjectName("inputPathL")
        self.gridLayout.addWidget(self.inputPathL, 0, 0, 1, 1)
        self.maskShapeL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.maskShapeL.setObjectName("maskShapeL")
        self.gridLayout.addWidget(self.maskShapeL, 1, 0, 1, 1)
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setSpacing(6)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.gridLayout.addLayout(self.gridLayout_5, 5, 1, 1, 1)
        self.outputPathHLayout = QtWidgets.QHBoxLayout()
        self.outputPathHLayout.setSpacing(6)
        self.outputPathHLayout.setObjectName("outputPathHLayout")
        self.outputPath = QtWidgets.QLineEdit(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.outputPath.sizePolicy().hasHeightForWidth())
        self.outputPath.setSizePolicy(sizePolicy)
        self.outputPath.setMaximumSize(QtCore.QSize(460, 16777215))
        self.outputPath.setDragEnabled(True)
        self.outputPath.setObjectName("outputPath")
        self.outputPathHLayout.addWidget(self.outputPath)
        self.saveBtn = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.saveBtn.setMaximumSize(QtCore.QSize(120, 16777215))
        self.saveBtn.setObjectName("saveBtn")
        self.outputPathHLayout.addWidget(self.saveBtn)
        self.outputPathEmptyL = QtWidgets.QLabel(self.gridLayoutWidget)
        self.outputPathEmptyL.setText("")
        self.outputPathEmptyL.setObjectName("outputPathEmptyL")
        self.outputPathHLayout.addWidget(self.outputPathEmptyL)
        self.gridLayout.addLayout(self.outputPathHLayout, 4, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        item = self.annTable.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "New Column"))
        self.outputPathL.setText(_translate("MainWindow", "output path"))
        self.radiLenL.setText(_translate("MainWindow", "radius  / length list of particles"))
        self.radiLen.setText(_translate("MainWindow", "6, 6, 3, 6, 6, 7, 6, 4, 4, 3, 10, 8"))
        self.showAnnL.setText(_translate("MainWindow", "Show annotations"))
        self.CircRBtn.setText(_translate("MainWindow", "Circle"))
        self.RecRBtn.setText(_translate("MainWindow", "Rectangle"))
        self.inputPath.setText(_translate("MainWindow", "../../data/HREC/0/micrographs/"))
        self.outputPath.setText(_translate("MainWindow", "C:/Users/Asus/Desktop/DeepTomo/data/SHREC/0/masks/"))
        self.loadBtn.setText(_translate("MainWindow", "load"))
        self.ShowAnnBtn.setText(_translate("MainWindow", "On"))
        self.inputPathL.setText(_translate("MainWindow", "Annotation file: (XML, CSV, txt, and star)"))
        self.maskShapeL.setText(_translate("MainWindow", "Mask Shape"))
        self.saveBtn.setText(_translate("MainWindow", "Save"))
