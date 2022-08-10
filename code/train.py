# ============================================================================================
# DeepET - a deep learning framework for segmentation and classification of
#                  macromolecules in Cryo Electron Tomograms (Cryo-ET)
# ============================================================================================
# Copyright (c) 2021 - now
# ZIB - Department of Visual and Data Centric
# Author: Noushin Hajarolasvadi
# Team Leader: Daniel Baum
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# ============================================================================================

from train_models import *
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QIcon, QColor
from gui.train import train
from gui.theme_style import *

from PyQt5.QtWidgets import QRadioButton, QHBoxLayout, QGridLayout, QButtonGroup
# from PyQt5.QtCore import pyqtSignal, QObject

class TrainingWindow(QMainWindow):
    def __init__(self):
        super(TrainingWindow, self).__init__()

        self.ui = train.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Training Models")
        self.setWindowIcon(QIcon('../../icon.jpg'))

        p = self.ui.textEdit.palette()
        p.setColor(QPalette.Active, QPalette.Base, Qt.black)
        self.ui.textEdit.setPalette(p)
        white_color = QColor(255, 255, 255)
        self.ui.textEdit.setTextColor(white_color)

        self.model_names = ["3D UNet", "3D UCaps", "TL 3D UNet"]
        self.loss_names = ["Dice", "BCE Dice", "Categorical", "Focal", "Focal tversky", "tversky"]
        self.opt_names = ["Adam", "SGD", "RMS Prop"]
        self.lr_names = ["Fixed", "Step", "Exponential", "Polynomial", "Cyclic"]

        self.generate_model_radio_btns(3)
        self.generate_optimizer_radio_btns(3)
        self.generate_loss_radio_btns(6)
        self.generate_lr_radio_btns(5)

        self.ui.trainBtn.clicked.connect(self.start_train)

        # initialize learning parameters
        self.base_path = None
        self.output_path = None
        self.epochs = None
        self.batch_size = None
        self.patch_size = None
        self.lr = None
        self.opt = None
        self.loss = None
        self.model_type = None
        self.lr_type = None
        self.dim_num = None
        self.img_dim = None
        self.class_names = None
        self.metrics = ["accuracy"]
        self.set_params(True)
        self.classNum = 0
        self.vald_prc = .2
        self.augm_prc = .3
        # self.get_model()

    def generate_model_radio_btns(self, number):
        horizontalLayoutModel = QHBoxLayout()
        horizontalBox = QGridLayout()

        horizontalLayoutModel.addLayout(horizontalBox)

        modelgroup_btns = QButtonGroup(self)
        modelgroup_btns.buttonClicked.connect(lambda btn: self.set_model(btn.text()))

        for btn_num in range(number):
            model_rbtn = QRadioButton()
            modelgroup_btns.addButton(model_rbtn)
            horizontalBox.addWidget(model_rbtn, 1, btn_num, 1, 1)
            model_rbtn.setText(self.model_names[btn_num])
            if btn_num == 0:
                model_rbtn.setChecked(True)

        self.ui.gridLayout_2.addLayout(horizontalLayoutModel, 8, 1, 1, 1)

    def generate_optimizer_radio_btns(self, number):
        horizontalLayoutOpt = QHBoxLayout()
        horizontalBox = QGridLayout()

        horizontalLayoutOpt.addLayout(horizontalBox)

        modelgroup_btns = QButtonGroup(self)
        modelgroup_btns.buttonClicked.connect(lambda btn: self.set_opt(btn.text()))

        for btn_num in range(number):
            model_rbtn = QRadioButton()
            modelgroup_btns.addButton(model_rbtn)
            horizontalBox.addWidget(model_rbtn, 1, btn_num, 1, 1)
            model_rbtn.setText(self.opt_names[btn_num])
            if btn_num == 0:
                model_rbtn.setChecked(True)

        self.ui.gridLayout_2.addLayout(horizontalLayoutOpt, 9, 1, 1, 1)

    def generate_loss_radio_btns(self, number):
        horizontalLayoutLoss = QHBoxLayout()
        horizontalBox = QGridLayout()

        horizontalLayoutLoss.addLayout(horizontalBox)

        modelgroup_btns = QButtonGroup(self)
        modelgroup_btns.buttonClicked.connect(lambda btn: self.set_loss(btn.text()))

        for btn_num in range(number):
            model_rbtn = QRadioButton()
            modelgroup_btns.addButton(model_rbtn)
            horizontalBox.addWidget(model_rbtn, 1, btn_num, 1, 1)
            model_rbtn.setText(self.loss_names[btn_num])
            if btn_num == 2:
                model_rbtn.setChecked(True)

        self.ui.gridLayout_2.addLayout(horizontalLayoutLoss, 10, 1, 1, 1)

    def generate_lr_radio_btns(self, number):
        horizontalLayoutLR = QHBoxLayout()
        horizontalBox = QGridLayout()

        horizontalLayoutLR.addLayout(horizontalBox)

        lrgroup_btns = QButtonGroup(self)
        lrgroup_btns.buttonClicked.connect(lambda btn: self.set_lr(btn.text()))

        for btn_num in range(number):
            lr_rbtn = QRadioButton()
            lrgroup_btns.addButton(lr_rbtn)
            horizontalBox.addWidget(lr_rbtn, 1, btn_num, 1, 1)
            lr_rbtn.setText(self.lr_names[btn_num])
            if btn_num == 0:
                lr_rbtn.setChecked(True)

        self.ui.gridLayout_2.addLayout(horizontalLayoutLR, 11, 1, 1, 1)


    def set_params(self, flag=True):
        # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        self.epochs = int(self.ui.epochs.text())
        self.batch_size = int(self.ui.batchsize.text())
        self.patch_size = int(self.ui.patchsize.text())
        self.base_path = ROOT_DIR.__str__() + self.ui.basePath.text()
        self.weight_path = ROOT_DIR.__str__() + self.ui.weightPath.text()
        self.output_path = ROOT_DIR.__str__() + str(self.ui.outputPath.text())
        self.lr = float(self.ui.LR.text())
        self.class_names = self.ui.classnames.text()
        self.classNum = len(self.class_names.split(","))+1
        self.vald_prc = float(self.ui.validation.text())
        self.augm_prc = float(self.ui.augment.text())

        if self.ui.depth == 0:
            self.dim_num = 2
            self.img_dim = (int(self.ui.width.text()), int(self.ui.height.text()))
        else:
            self.dim_num = 3
            self.img_dim = (int(self.ui.width.text()), int(self.ui.height.text()), int(self.ui.depth.text()))

        # ToDo: if you want to have particular learning rates
        if flag:
            self.set_model(self.model_names[0])
            self.set_loss(self.loss_names[2])
            self.set_opt(self.opt_names[0])
            self.set_lr(self.lr_names[0])

    def set_model(self, radio_text):
        self.model_type = radio_text

    def set_lr(self, radio_text):
        self.lr_type = radio_text

    def set_opt(self, radio_text):
        self.opt = radio_text

    def set_loss(self, radio_text):
        if radio_text == "Dice":
            self.loss = "dice_loss"
        elif radio_text == "BCE Dice":
            self.loss = "bce_dice_loss"
        elif radio_text == "Categorical":
            self.loss = "categorical_crossentropy"
        elif radio_text == "Focal":
            self.loss = "focal_loss"
        elif radio_text == "Focal tversky":
            self.loss = "focal_tversky"
        elif radio_text == "tversky":
            self.loss = "tversky"

    def start_train(self):
        self.set_params(False)
        model_obj = TrainModel(self)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    set_theme_style(app)

    application = TrainingWindow()
    application.show()

    sys.exit(app.exec_())
