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
from gui.theme_style import *
from utils.params import *


class TrainingWindow(QMainWindow):
    def __init__(self):
        super(TrainingWindow, self).__init__()

        self.ui = train.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Training Models")
        self.setWindowIcon(QIcon('../../icon.jpg'))

        # inititalize some parameters
        self.batch_size = 4
        self.patch_size = 64
        self.lr = 0.01
        self.opt = "Adam"

    # def set_params(self):

if __name__ == "__main__":
    app = QApplication(sys.argv)
    set_theme_style(app)

    application = TrainingWindow()
    application.show()

    sys.exit(app.exec_())
