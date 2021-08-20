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
import os.path

from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow
from PyQt5.QtGui import QIcon
from gui.evaluation import evaluation
from gui.theme_style import *
from utils.params import *
from models import *
from keras.models import load_model
from PIL import Image as pilimg
from keras.preprocessing import image


class EvaluationWindow(QMainWindow):
    def __init__(self):
        super(EvaluationWindow, self).__init__()

        self.ui = evaluation.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Evaluate a Model")
        self.setWindowIcon(QIcon('../../icon.jpg'))

        self.ui.evalBtn.clicked.connect(self.start_evaluation)

        self.patch_size = 64
        self.tomo_shape = None
        self.model_type = None
        self.model = None

    def set_model(self, radio_text):
        self.model_type = radio_text

    def start_evaluation(self):
        # laod the model
        tomo_path = ROOT_DIR.__str__() + str(self.ui.input_path.text())
        model_path = ROOT_DIR.__str__() + str(self.ui.model_path.text())

        tomo = read_mrc(os.path.join(tomo_path, "grandmodel_9.mrc"))
        # mask = read_mrc(os.path.join(tomo_path, "target_grandmodel_9.mrc"))

        # c = np.int(np.floor(self.patch_size / 2))
        self.tomo_shape = tomo.shape
        self.model = load_model(model_path)

        # list_annotations = self.get_coordinates()
        patches_tomo = self.extract_patches(tomo)
        self.predictions(patches_tomo)


    def extract_patches(self, tomo):
        display("fetching patches...")
        tomo = (tomo - np.mean(tomo)) / np.std(tomo)

        # preparation of tomogram as a tensor that can be used with tensorflow API
        tomo = np.swapaxes(tomo, 0, 2)  # changing dimension order from (z, y, x) to (x, y, z)
        tomo = np.expand_dims(tomo, axis=0)  # expanding dimensions for tensorflow input
        tomo = np.expand_dims(tomo, axis=4)  # expanding dimensions for tensorflow input

        # extracting patches of size patch_size * patch_size * patch_size
        patches_tomo = tf.extract_volume_patches(tomo, [1, self.patch_size, self.patch_size, self.patch_size, 1],
                                                 [1, self.patch_size, self.patch_size, self.patch_size, 1],
                                                 padding='VALID')
        patches_tomo = tf.reshape(patches_tomo, [-1, self.patch_size, self.patch_size, self.patch_size])
        patches_tomo = tf.squeeze(patches_tomo)

        # converting back from tensor to numpy
        patches_tomo = patches_tomo.eval(session=tf.compat.v1.Session())

        # the images are
        patches_tomo = np.swapaxes(patches_tomo, 1, 3)  # changing back dimension order from (x, y, z) to (z, y, x)
        print(patches_tomo.shape)
        print("Finished")
        return patches_tomo

    def predictions(self, patches_tomo):
        tomo_seg = np.zeros(self.tomo_shape)
        for i in range(patches_tomo.shape[0]):
            tomo = patches_tomo[i]
            tomo = np.reshape(tomo, (1, self.patch_size, self.patch_size, self.patch_size, 1))
            predicted_vals = self.model.predict(tomo)
            predicted_vals = np.argmax(predicted_vals, axis=-1)

    def get_coordinates(self):
        # TODO: it is better to get the filename from user!
        xml_path = ROOT_DIR.__str__() + str(self.ui.input_path.text())
        list_annotations = read_xml2(os.path.join(xml_path, "images/object_list_test.xml"))

        return list_annotations

    def display_mask(self, target):
        center_slide = np.int(np.floor(target.shape[0] / 2))
        """Quick utility to display a model's prediction."""
        slice = target[center_slide-1:center_slide+1][:][:]
        slice = np.moveaxis(slice, 0, -1)

        plt.imshow(slice, vmin=np.min(slice), vmax=np.max(slice))

        # converting MRC to png
        # slice = (255.0 / slice.max() * (slice - slice.min())).astype(np.uint8)
        # slice = pilimg.fromarray(slice)
        # if slice.mode != 'RGB':
        #     slice = slice.convert('RGB')
        # slice.save("myet.png")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    set_theme_style(app)

    application = EvaluationWindow()
    application.show()

    sys.exit(app.exec_())
