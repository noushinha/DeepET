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
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QIcon
from gui.reg_evaluation import reg_evaluation
from gui.theme_style import *
from utils.params import *
from train_models import *
from PyQt5.QtWidgets import QRadioButton, QHBoxLayout, QGridLayout, QButtonGroup
from sklearn.metrics import r2_score

class EvaluationWindow(QMainWindow):
    def __init__(self):
        super(EvaluationWindow, self).__init__()

        self.ui = reg_evaluation.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Evaluate a Model")
        self.setWindowIcon(QIcon('../../icon.jpg'))

        self.ui.evalBtn.clicked.connect(self.calculate_r2score)
        self.model_names = ["3D UNet", "3D UCAP"]  # "YOLOv3", "R-CNN", "Mask R-CNN"]
        self.generate_model_radio_btns(2)

        self.patch_size = 56
        self.patch_overlap = (self.patch_size / 2)
        self.slide = None
        self.tomo = None
        self.model_type = "3D UNet"
        self.model_path = None
        self.output_path = None
        self.model = None
        self.tomo_path = None

        self.tomo_id = 9
        self.class_id = "RB_distance_map"

        # initialize the parameters
        self.set_params(True)

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

        self.ui.horizontalLayout_3.addLayout(horizontalLayoutModel, 3)

    def set_model(self, radio_text):
        self.model_type = radio_text

    def set_params(self, flag=True):
        self.tomo_path = ROOT_DIR.__str__() + str(self.ui.input_path.text())
        self.model_path = ROOT_DIR.__str__() + str(self.ui.model_path.text())
        self.output_path = ROOT_DIR.__str__() + str(self.ui.output_path.text())
        self.mask_path = str.replace(self.tomo_path, "tomo", "distancemap")

        is_file(self.tomo_path)
        is_file(self.mask_path)
        is_file(self.model_path)
        is_dir(self.output_path)

        self.tomo = read_mrc(self.tomo_path)
        # self.tomo = np.flip(self.tomo)
        self.mask = read_mrc(self.mask_path)
        # self.mask = np.flip(self.mask)
        # self.tomo = nib.load(self.tomo_path)
        # self.tomo = self.tomo.get_fdata()

        self.cnn_model()

        if flag:
            self.set_model(self.model_names[0])

    def cnn_model(self):

        if self.model_type == "3D UNet":
            cnnobj = CNNModels()
            self.model = cnnobj.unet3d((self.patch_size, self.patch_size, self.patch_size), 2)
            self.model.load_weights(self.model_path)

    def calculate_r2score(self):
        """
        calculates the r2 score for regression problems
        we need the ground truth versus predictions as a 1D vector
        :return: single scalar value as a r2score
        """
        tomo = (self.tomo - np.mean(self.tomo)) / np.std(self.tomo)
        tomo = np.swapaxes(tomo, 0, 2)  # changing dimension order from (z, y, x) to (x, y, z)
        tomo = np.expand_dims(tomo, axis=0)  # expanding dimensions for tensorflow input
        tomo = np.expand_dims(tomo, axis=4)  # expanding dimensions for tensorflow input
        print(tomo.shape)


        mask = (self.mask - np.mean(self.mask)) / np.std(self.mask)
        mask = np.swapaxes(mask, 0, 2)  # changing dimension order from (z, y, x) to (x, y, z)
        mask = np.expand_dims(mask, axis=0)  # expanding dimensions for tensorflow input
        mask = np.expand_dims(mask, axis=4)  # expanding dimensions for tensorflow input
        print(mask.shape)

        strd_win = int(self.patch_size / 2)
        ksizes = [1, self.patch_size, self.patch_size, self.patch_size, 1]
        strides = [1, self.patch_size, self.patch_size, self.patch_size, 1]

        extracted_tomo_patches = tf.extract_volume_patches(tomo, ksizes, strides, "VALID", name=None)
        extracted_mask_patches = tf.extract_volume_patches(mask, ksizes, strides, "VALID", name=None)
        print("Tomo patches shape: ", extracted_tomo_patches.shape)
        print("Mask patches shape: ", extracted_mask_patches.shape)

        tomo_patches = next(iter(extracted_tomo_patches))
        mask_patches = next(iter(extracted_mask_patches))

        cnt = 0
        for depth in range(tomo_patches.shape[2]):  # in z direction
            for col in range(tomo_patches.shape[1]):  # in y directino
                for row in range(tomo_patches.shape[0]):  # in x direction
                    tomo_patch = tomo_patches[row, col, depth]
                    tomo_patch = tf.reshape(tomo_patch, [self.patch_size, self.patch_size, self.patch_size]).numpy()
                    tomo_patch = np.expand_dims(tomo_patch, axis=0)

                    pred_mask_patch = self.model.predict(tomo_patch, batch_size=1)

                    mask_patch = mask_patches[row, col, depth]
                    mask_patch2 = tf.reshape(mask_patch, [self.patch_size, self.patch_size, self.patch_size]).numpy()

                    # resave_tomo = tomo_patch[0, :, :, :]
                    # resave_tomo = np.swapaxes(resave_tomo, 0, 2)
                    # resave_mask = np.swapaxes(mask_patch2, 0, 2)
                    # prd_mask = pred_mask_patch[0, :, :, :, 0]
                    # prd_mask = np.swapaxes(prd_mask, 0, 2)
                    #
                    # write_mrc2(np.flip(resave_tomo, axis=1),
                    #            f"/media/noushin/Data/Cryo-ET/DeepET/data2/temp/test_patch_extraction/tomo_resaved_patch_{cnt}.mrc")
                    # write_mrc2(np.flip(resave_mask, axis=1),
                    #            f"/media/noushin/Data/Cryo-ET/DeepET/data2/temp/test_patch_extraction/mask_resaved_patch_{cnt}.mrc")
                    # write_mrc2(np.flip(prd_mask, axis=1),
                    #            f"/media/noushin/Data/Cryo-ET/DeepET/data2/temp/test_patch_extraction/prd_mask_resaved_patch_{cnt}.mrc")



                    # score = r2_score(mask_patch.numpy().flatten(),
                    #                  pred_mask_patch.flatten())
                    # print(score)
                    cnt += 1


        # for i in extracted_patches.shape[0]:
        #     print("item: ", i)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    set_theme_style(app)

    application = EvaluationWindow()
    application.show()

    sys.exit(app.exec_())
