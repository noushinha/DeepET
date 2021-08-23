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

import numpy as np
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
        self.num_class = 13
        self.slide = self.patch_size
        self.tomo_shape = None
        self.model_type = None
        self.model = None
        self.tomo_path = None

    def set_model(self, radio_text):
        self.model_type = radio_text

    def start_evaluation(self):
        # laod the model
        self.tomo_path = ROOT_DIR.__str__() + str(self.ui.input_path.text())
        model_path = ROOT_DIR.__str__() + str(self.ui.model_path.text())

        tomo = read_mrc(os.path.join(self.tomo_path, "grandmodel_9.mrc"))

        self.tomo_shape = tomo.shape
        self.model = load_model(model_path)

        scores_tomo = self.extract_patches(tomo)
        labels_tomo = np.int8(np.argmax(scores_tomo, 3)) #convert scoremaps to class label

        self.save_result(scores_tomo, labels_tomo)

    def extract_patches(self, tomo):
        display("fetching patches...")
        hdim = int(self.patch_size / 2)
        slide = self.patch_size + 1
        # z_half_dim, y_half_dim, x_half_dim = (self.patch_size / 2, self.patch_size / 2, self.patch_size / 2)
        x_centers = list(range(hdim, tomo.shape[2] - hdim, slide))
        y_centers = list(range(hdim, tomo.shape[1] - hdim, slide))
        z_centers = list(range(hdim, tomo.shape[0] - hdim, slide))

        # if dimensions are not exactly divisible,
        # we should collect the remained voxels around borders
        x_centers, y_centers, z_centers = correct_center_positions(x_centers, y_centers, z_centers, tomo.shape, hdim)
        # total number of patches that we should extract
        total_pnum = len(x_centers) * len(y_centers) * len(z_centers)

        # normalize and zero pad the tomogram values
        tomo = (tomo - np.mean(tomo)) / np.std(tomo)
        tomo = np.pad(tomo, 0, mode='constant', constant_values=0)
        # two arrays to collect ther esults of rpediction
        # one for predicted intensity values, second hodls the predicted class labels
        pred_tvals = np.zeros(self.tomo_shape).astype(np.int8)
        pred_tclass = np.zeros(self.tomo_shape + (self.num_class,)).astype(np.float16)  # tomo.shape * # classes

        patch_num = 1
        for z in z_centers:
            for y in y_centers:
                for x in x_centers:
                    display('patch number ' + str(patch_num) + ' out of ' + str(total_pnum))
                    patch = tomo[z-hdim:z+hdim, y-hdim:y+hdim, x-hdim:x+hdim]
                    patch = np.expand_dims(patch, axis=0)  # expanding dimensions for predict function
                    patch = np.expand_dims(patch, axis=4)  # expanding dimensions for predict function
                    pred_vals = self.model.predict(patch, batch_size=1)

                    # predicted_classes = np.argmax(predicted_vals, axis=-1)
                    current_patch = pred_tclass[z-hdim:z+hdim, y-hdim:y+hdim, x-hdim:x+hdim, :]
                    casted_pred_vals = np.float16(pred_vals[0, 0:2*hdim, 0:2*hdim, 0:2*hdim, :])
                    pred_tclass[z-hdim:z+hdim, y-hdim:y+hdim, x-hdim:x+hdim] = current_patch + casted_pred_vals

                    current_patch2 = pred_tvals[z-hdim:z+hdim, y-hdim:y+hdim, x-hdim:x+hdim]
                    argmax_labels = np.ones((self.patch_size, self.patch_size, self.patch_size), dtype=np.int8)
                    pred_tvals[z-hdim:z+hdim, y-hdim:y+hdim, x-hdim:x+hdim] = current_patch2 + argmax_labels

                    patch_num += 1

        # print(patches_tomo.shape)
        print("Fetching Finished")

        # required only if there are overlapping regions (normalization)
        for n in range(self.num_class):
            pred_tclass[:, :, :, n] = pred_tclass[:, :, :, n] / pred_tvals

        return pred_tclass



    def save_result(self, scoremap_tomo, labelmap_tomo):
        binned_scoremap = self.bin_tomo(scoremap_tomo)
        binned_labelmap = np.int8(np.argmax(binned_scoremap, 3))

        # Save labelmaps:
        write_mrc(labelmap_tomo, os.path.join(self.tomo_path, 'tomo_labelmap.mrc'))
        write_mrc(binned_labelmap, os.path.join(self.tomo_path, 'tomo_binned_labelmap.mrc'))

    def bin_tomo(self, scoremap_tomo):
        from skimage.measure import block_reduce

        bd0 = np.int(np.ceil(scoremap_tomo.shape[0] / 2))
        bd1 = np.int(np.ceil(scoremap_tomo.shape[1] / 2))
        bd2 = np.int(np.ceil(scoremap_tomo.shape[2] / 2))
        new_dim = (bd0, bd1, bd2, self.num_class)
        binned_scoremap = np.zeros(new_dim)

        for cnum in range(self.num_class):
            binned_scoremap[:, :, :, cnum] = block_reduce(scoremap_tomo[:, :, :, cnum], (2, 2, 2), np.mean)

        return binned_scoremap


if __name__ == "__main__":
    app = QApplication(sys.argv)
    set_theme_style(app)

    application = EvaluationWindow()
    application.show()

    sys.exit(app.exec_())
