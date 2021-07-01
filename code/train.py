# ============================================================================================
# DeepTomo - a deep learning framework for segmentation and classification of
#                  macromolecules in Cryo Electron Tomograms (Cryo-ET)
# ============================================================================================
# Copyright (c) 2021 - now
# ZIB - Department of Visual and Data Centric
# Author: Noushin Hajarolasvadi, Willy (Daniel team)
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# ============================================================================================

from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem, QApplication, QMainWindow
from PyQt5.QtGui import QIcon
from gui.train import train
from keras.optimizers import Adam, Adamax, RMSprop, SGD
from gui.theme_style import *
from utils.params import *
from models import *


class TrainingWindow(QMainWindow):
    def __init__(self):
        super(TrainingWindow, self).__init__()

        self.ui = train.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Training Models")
        self.setWindowIcon(QIcon('../../icon.jpg'))

        # inititalize some parameters
        self.epochs = 100
        self.batch_size = 20
        self.patch_size = 64
        self.lr = 0.01
        self.opt = "Adam"
        self.optimizer = None
        self.loss = "binary"

        self.model = CNNModels.unet3d()
        self.set_optimizer()
        self.set_loss()
        self.set_lr()
        self.save_weights()

    def set_optimizer(self):
        if self.opt == "RMS":
            self.optimizer = RMSprop(lr=self.lr, rho=0.9, epsilon=1e-06, clipnorm=0, clipvalue=10)
        elif self.opt == "SGD":
            self.optimizer = SGD(lr=self.lr, momentum=0.0, decay=0.0, nesterov=False, clipnorm=0, clipvalue=10)
        elif self.opt == "Adam":
            self.optimizer = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=0, clipvalue=10)
        elif self.opt == "Ada":
            self.optimizer = Adamax(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=0, clipvalue=10)

    def set_loss(self):
        if self.loss == "binary":
            return "binary_crossentropy"
        elif self.loss == "multi":
            return "categorical_crossentropy"

    def get_lr(self):
        return 1
    # def set_params(self):

if __name__ == "__main__":
    app = QApplication(sys.argv)
    set_theme_style(app)

    application = TrainingWindow()
    application.show()

    sys.exit(app.exec_())
