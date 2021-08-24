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


from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QIcon
from gui.evaluation import evaluation
from gui.theme_style import *
from utils.params import *
from models import *
from keras.models import load_model
from sklearn.cluster import MeanShift
from PyQt5.QtWidgets import QRadioButton, QHBoxLayout, QGridLayout, QButtonGroup

class EvaluationWindow(QMainWindow):
    def __init__(self):
        super(EvaluationWindow, self).__init__()

        self.ui = evaluation.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Evaluate a Model")
        self.setWindowIcon(QIcon('../../icon.jpg'))

        self.ui.evalBtn.clicked.connect(self.start_evaluation)
        self.model_names = ["3D UNet", "YOLOv3", "R-CNN", "Mask R-CNN"]
        self.generate_model_radio_btns(4)

        self.patch_size = 64
        self.num_class = 13
        self.slide = self.patch_size
        self.tomo_shape = None
        self.model_type = None
        self.model = None
        self.tomo_path = None
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
        if flag:
            self.set_model(self.model_names[0])
    def start_evaluation(self):
        """
        We segment the tomogram to non-overlapped patches
        Args: None (requires just the path to tomogram and the network weights)
        :return: numpy array of scoremap predicted by the trained model
        """
        # laod the model
        self.set_params(False)
        self.tomo_path = ROOT_DIR.__str__() + str(self.ui.input_path.text())
        model_path = ROOT_DIR.__str__() + str(self.ui.model_path.text())

        tomo = read_mrc(os.path.join(self.tomo_path, "grandmodel_9.mrc"))

        self.tomo_shape = tomo.shape
        # self.model = load_model(model_path)

        cnnobj = CNNModels()
        if self.model_type == "3D UNet":
            self.model = cnnobj.unet3d((self.patch_size, self.patch_size, self.patch_size), self.num_class)
            self.model.load_weights(model_path)
        # scores_tomo = self.extract_patches(tomo)
        # labels_tomo = np.int8(np.argmax(scores_tomo, axis=-1))  # convert scoremaps to class label
        #
        # # save labelmaps
        # binned_labelmap = self.save_result(scores_tomo, labels_tomo)
        # print(np.unique(binned_labelmap))
        # # plot labelmaps
        # plot_vol(labels_tomo, self.tomo_path)
        radi = 5
        thr = 1
        binned_labelmap = read_mrc("C:\\Users\Asus\Desktop\DeepET\data2\\tomo_binned_labelmap.mrc")
        self.save_coordinates(binned_labelmap, radi, thr)


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
                    # patch = np.reshape(patch, (1, self.patch_size, self.patch_size, self.patch_size, 1))
                    patch = np.expand_dims(patch, axis=0)  # expanding dimensions for predict function
                    patch = np.expand_dims(patch, axis=4)  # expanding dimensions for predict function
                    pred_vals = self.model.predict(patch, batch_size=1)
                    # print(np.unique(np.argmax(pred_vals, 4)))
                    current_patch = pred_tclass[z-hdim:z+hdim, y-hdim:y+hdim, x-hdim:x+hdim, :]
                    casted_pred_vals = np.float16(pred_vals[0, 0:2*hdim, 0:2*hdim, 0:2*hdim, :])
                    pred_tclass[z-hdim:z+hdim, y-hdim:y+hdim, x-hdim:x+hdim] = current_patch + casted_pred_vals

                    current_patch2 = pred_tvals[z-hdim:z+hdim, y-hdim:y+hdim, x-hdim:x+hdim]
                    argmax_labels = np.ones((self.patch_size, self.patch_size, self.patch_size), dtype=np.int8)
                    pred_tvals[z-hdim:z+hdim, y-hdim:y+hdim, x-hdim:x+hdim] = current_patch2 + argmax_labels

                    patch_num += 1

        # print(patches_tomo.shape)
        print("Fetching Finished")
        print(np.unique(np.argmax(pred_tclass, 3)))
        # required only if there are overlapping regions (normalization)
        for n in range(self.num_class):
            pred_tclass[:, :, :, n] = pred_tclass[:, :, :, n] / pred_tvals
        print(np.unique(np.argmax(pred_tclass, 3)))
        return pred_tclass

    def save_result(self, scoremap_tomo, labelmap_tomo):
        binned_scoremap = self.bin_tomo(scoremap_tomo)
        binned_labelmap = np.int8(np.argmax(binned_scoremap, axis=-1))

        # Save labelmaps:
        write_mrc(labelmap_tomo, os.path.join(self.tomo_path, 'tomo_labelmap.mrc'))
        write_mrc(binned_labelmap, os.path.join(self.tomo_path, 'tomo_binned_labelmap.mrc'))
        return binned_labelmap

    def bin_tomo(self, scoremap_tomo):
        from skimage.measure import block_reduce

        bd0 = int(np.ceil(scoremap_tomo.shape[0] / 2))
        bd1 = int(np.ceil(scoremap_tomo.shape[1] / 2))
        bd2 = int(np.ceil(scoremap_tomo.shape[2] / 2))
        new_dim = (bd0, bd1, bd2, self.num_class)
        binned_scoremap = np.zeros(new_dim)

        for cnum in range(self.num_class):
            binned_scoremap[:, :, :, cnum] = block_reduce(scoremap_tomo[:, :, :, cnum], (2, 2, 2), np.mean)

        return binned_scoremap

    def save_coordinates(self, binned_labelmap, radi, thr):
        """
        This function returns coordinates of individual particles. Meanshift clustering is used.
        :param binned_labelmap_tomo: the labelmap to be segmented and analysed
        :param radi: cluster radius
        :param thr: threshold for discarding based on radius
        :return: list of coordinates and corresponding classes of the clustered particles
        """
        # testvals = binned_labelmap[binned_labelmap > 0]
        obj_vals = np.nonzero(binned_labelmap > 0)
        print(np.unique(binned_labelmap))
        obj_vals = np.array(obj_vals).T
        clusters = MeanShift(bandwidth=radi, bin_seeding=True).fit(obj_vals)
        num_clusters = clusters.cluster_centers_.shape[0]

        object_list = []
        labels = np.zeros((12,))
        for n in range(num_clusters):
            cluster_data_point_indx = np.nonzero(clusters.labels_ == n)

            # Get cluster size and position:
            cluster_size = np.size(cluster_data_point_indx)
            centroids = clusters.cluster_centers_[n]

            # Attribute a macromolecule class to cluster:
            cluster_elements = []
            for c in range(cluster_size):  # get labels of cluster members
                element_coord = obj_vals[cluster_data_point_indx[0][c], :]
                cluster_elements.append(binned_labelmap[element_coord[0], element_coord[1], element_coord[2]])

            for num in range(12):  # get most present label in cluster
                labels[num] = np.size(np.nonzero(np.array(cluster_elements) == num + 1))
            assigned_label = np.argmax(labels) + 1

            object_list = add_obj(object_list, label=assigned_label, coord=centroids, c_size=cluster_size)

        self.print_stats(object_list)
        return object_list

    def print_stats(self, alist):
        display('----------------------------------------')
        display('A total of ' + str(len(alist)) + ' objects has been found.')

        classes = []
        for idx in range(len(alist)):
            classes.append(alist[idx]['label'])
        lbl_set = set(classes)
        lbls = (list(lbl_set))

        for l in lbls:
            class_id = []
            for i in range(len(alist)):
                if str(alist[i]['label']) == str(l):
                    class_id.append(i)

            obj_class_list = []
            for id in range(len(class_id)):
                obj_class_list.append(alist[class_id[id]])

            display('Class ' + str(l) + ': ' + str(len(obj_class_list)) + ' objects')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    set_theme_style(app)

    application = EvaluationWindow()
    application.show()

    sys.exit(app.exec_())
