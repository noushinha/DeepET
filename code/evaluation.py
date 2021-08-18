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
        self.model_type = None
        self.model = None

    def set_model(self, radio_text):
        self.model_type = radio_text

    def start_evaluation(self):
        # laod the model
        tomo_path = ROOT_DIR.__str__() + str(self.ui.input_path.text())
        model_path = ROOT_DIR.__str__() + str(self.ui.model_path.text())
        c = np.int(np.floor(self.patch_size / 2))

        self.model = load_model(model_path)

        tomo = read_mrc(os.path.join(tomo_path, "images/tomo.mrc"))
        mask = read_mrc(os.path.join(tomo_path, "targets/target.mrc"))

        list_annotations = self.get_coordinates()
        self.extract_patches(tomo, mask, list_annotations)


        # accuracy, loss, F1-Score


    def extract_patches(self, tomo, mask, list_annotations):
        display("fetching the tomogram...")

        start = time.clock()
        c = np.int(np.floor(self.patch_size / 2))
        num_annotations = len(list_annotations)
        tomo_target = np.zeros(tomo.shape)

        for i in range(num_annotations):
            tomo_idx = int(list_annotations[i]['tomo_idx'])

            # Get patch position:
            x, y, z = get_patch_position(tomo.shape, c, list_annotations[i], 13)
            # extract the patch:
            patch_tomo = tomo[z - c:z + c, y - c:y + c, x - c:x + c]
            patch_tomo = (patch_tomo - np.mean(patch_tomo)) / np.std(patch_tomo)

            patch_mask = mask[z - c:z + c, y - c:y + c, x - c:x + c]
            patch_mask_onehot = to_categorical(patch_mask, 13)
            loss_val = self.model.evaluate(patch_tomo, patch_mask_onehot, verbose=0)

            predicted_vals = self.model.predict(patch_tomo)
            predicted_vals = np.argmax(predicted_vals[i], axis=-1)
            # mask = np.expand_dims(mask, axis=-1)
            x, y, z = get_patch_position(tomo.shape, self.patch_size, list_annotations[i], 13)
            tomo_target[z - c:z + c, y - c:y + c, x - c:x + c] = predicted_vals

        tomo_path = ROOT_DIR.__str__() + str(self.ui.input_path.text())
        self.display_mask(tomo_target)
        write_mrc(tomo_target, os.path.join(tomo_path, "tomo_target.mrc"))

        end = time.clock()
        process_time = (end - start)
        display("patches fetched in {:.2f} seconds.".format(round(process_time, 2)))


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
