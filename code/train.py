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

from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem, QApplication, QMainWindow
from PyQt5.QtGui import QIcon
from gui.train import train
from keras.optimizers import Adam, Adamax, RMSprop, SGD
from gui.theme_style import *
from utils.params import *
from models import *
from PyQt5.QtWidgets import QRadioButton, QHBoxLayout, QGridLayout, QButtonGroup
from PyQt5.QtCore import pyqtSignal, QObject


class TrainingWindow(QMainWindow):
    def __init__(self):
        super(TrainingWindow, self).__init__()

        self.ui = train.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Training Models")
        self.setWindowIcon(QIcon('../../icon.jpg'))

        self.generate_model_radio_btns(4)
        self.generate_optimizer_radio_btns(4)
        self.generate_loss_radio_btns(2)

        self.ui.trainBtn.clicked.connect(self.start_train)

        # initialize learning parameters
        self.img_path = None
        self.target_path = None
        self.output_path = None
        self.epochs = None
        self.batch_size = None
        self.patch_size = None
        self.lr = None
        self.opt = None
        self.loss = None
        self.model = None
        self.dim_num = None
        self.img_dim = None
        self.set_params()
        # self.get_model()

    def generate_model_radio_btns(self, number):
        model_names= ["2D UNet", "3D UNet", "RPN", "Mask R-CNN"]

        horizontalLayoutModel = QHBoxLayout()
        horizontalBox = QGridLayout()

        horizontalLayoutModel.addLayout(horizontalBox)

        modelgroup_btns = QButtonGroup(self)
        modelgroup_btns.buttonClicked.connect(lambda btn: self.set_model(btn.text()))

        for btn_num in range(number):
            model_rbtn = QRadioButton()
            modelgroup_btns.addButton(model_rbtn)
            horizontalBox.addWidget(model_rbtn, 1, btn_num, 1, 1)
            model_rbtn.setText(model_names[btn_num])
            if btn_num == 0:
                model_rbtn.setChecked(True)

        self.ui.gridLayout_2.addLayout(horizontalLayoutModel, 3, 1, 1, 1)

    def generate_loss_radio_btns(self, number):
        model_names = ["Binary", "Categorical"]

        horizontalLayoutLoss = QHBoxLayout()
        horizontalBox = QGridLayout()

        horizontalLayoutLoss.addLayout(horizontalBox)

        modelgroup_btns = QButtonGroup(self)
        modelgroup_btns.buttonClicked.connect(lambda btn: self.set_loss(btn.text()))

        for btn_num in range(number):
            model_rbtn = QRadioButton()
            modelgroup_btns.addButton(model_rbtn)
            horizontalBox.addWidget(model_rbtn, 1, btn_num, 1, 1)
            model_rbtn.setText(model_names[btn_num])
            if btn_num == 0:
                model_rbtn.setChecked(True)

        self.ui.gridLayout_2.addLayout(horizontalLayoutLoss, 7, 1, 1, 1)

    def generate_optimizer_radio_btns(self, number):
        model_names = ["Adam", "AdaMaX", "SGD", "RMS Prop"]

        horizontalLayoutOpt = QHBoxLayout()
        horizontalBox = QGridLayout()

        horizontalLayoutOpt.addLayout(horizontalBox)

        modelgroup_btns = QButtonGroup(self)
        modelgroup_btns.buttonClicked.connect(lambda btn: self.set_opt(btn.text()))

        for btn_num in range(number):
            model_rbtn = QRadioButton()
            modelgroup_btns.addButton(model_rbtn)
            horizontalBox.addWidget(model_rbtn, 1, btn_num, 1, 1)
            model_rbtn.setText(model_names[btn_num])
            if btn_num == 0:
                model_rbtn.setChecked(True)

        self.ui.gridLayout_2.addLayout(horizontalLayoutOpt, 5, 1, 1, 1)

    def set_params(self):
        self.epochs = int(self.ui.epochs.text())
        self.batch_size = int(self.ui.batchsize.text())
        self.patch_size = int(self.ui.patchsize.text())
        self.img_path = self.ui.imagePath.text()
        self.target_path = self.ui.targetPath.text()
        self.output_path = self.ui.outputPath.text()
        self.lr = float(self.ui.LR.text())

        if self.ui.depth == 0:
            self.dim_num = 2
            self.img_dim = (self.ui.width, self.ui.height)
        else:
            self.dim_num = 3
            self.img_dim = (self.ui.width, self.ui.height, self.ui.depth)

        # ToDo: if you want to have particular learning rates
        # self.set_lr()

    def set_model(self, radio_text):
        self.model = radio_text

    def set_opt(self, radio_text):
        self.opt = radio_text

    def set_loss(self, radio_text):
        if radio_text == "Binary":
            self.loss = "binary_crossentropy"
        elif radio_text == "Categorical":
            self.loss = "categorical_crossentropy"

    def start_train(self):
        model_obj = CNNModels(self)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    set_theme_style(app)

    application = TrainingWindow()
    application.show()

    sys.exit(app.exec_())
