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

from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow
from PyQt5.QtGui import QIcon
from gui.evaluation import evaluation
from gui.theme_style import *
from utils.params import *
from models import *
from PyQt5.QtCore import pyqtSignal, QObject
from keras.models import load_model


class EvaluationWindow(QMainWindow):
    def __init__(self):
        super(EvaluationWindow, self).__init__()

        self.ui = evaluation.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Evaluate a Model")
        self.setWindowIcon(QIcon('../../icon.jpg'))

        self.ui.evalBtn.clicked.connect(self.start_evaluation)

        self.model_type = None

    def set_model(self, radio_text):
        self.model_type = radio_text

    def start_evaluation(self):
        # laod the model
        self.model_path = ROOT_DIR.__str__() + str(self.ui.model_path.text())
        model = load_model(self.model_path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    set_theme_style(app)

    application = EvaluationWindow()
    application.show()

    sys.exit(app.exec_())
