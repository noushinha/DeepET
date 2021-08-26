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
from copy import deepcopy

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

        self.ui.segtBtn.clicked.connect(self.start_segmentation)
        self.ui.clusBtn.clicked.connect(self.start_clustering)
        self.ui.evalBtn.clicked.connect(self.start_evaluation)
        self.model_names = ["3D UNet", "YOLOv3", "R-CNN", "Mask R-CNN"]
        self.generate_model_radio_btns(4)

        self.patch_size = 64
        self.num_class = 13
        self.slide = self.patch_size
        self.tomo = None
        self.model_type = "3D UNet"
        self.model_path = None
        self.output_path = None
        self.model = None
        self.tomo_path = None

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

        is_file(self.tomo_path)
        is_file(self.model_path)
        is_dir(self.output_path)

        self.tomo = read_mrc(self.tomo_path)
        self.cnn_model()

        if flag:
            self.set_model(self.model_names[0])

    def cnn_model(self):
        # self.model = load_model(self.model_path)

        cnnobj = CNNModels()
        if self.model_type == "3D UNet":
            self.model = cnnobj.unet3d((self.patch_size, self.patch_size, self.patch_size), self.num_class)
            self.model.load_weights(self.model_path)

    def start_segmentation(self):
        """
        We segment the tomogram to non-overlapped patches
        Args: None (requires just the path to tomogram and the network weights)
        :return: numpy array of scoremap predicted by the trained model
        """
        # laod the model and set the required params
        self.set_params(False)

        # extract patches from the tomo
        scores_tomo = self.extract_patches()

        # convert confidence values to class labels to generate a label map
        labels_tomo = np.int8(np.argmax(scores_tomo, axis=-1))

        # save the label map
        self.save_result(scores_tomo, labels_tomo)

        # plot the label map
        plot_vol(labels_tomo, self.output_path)

    def extract_patches(self):
        display("fetching patches...")
        hdim = int(self.patch_size / 2)
        slide = self.patch_size + 1
        # z_half_dim, y_half_dim, x_half_dim = (self.patch_size / 2, self.patch_size / 2, self.patch_size / 2)
        x_centers = list(range(hdim, self.tomo.shape[2] - hdim, slide))
        y_centers = list(range(hdim, self.tomo.shape[1] - hdim, slide))
        z_centers = list(range(hdim, self.tomo.shape[0] - hdim, slide))

        # if dimensions are not exactly divisible,
        # we should collect the remained voxels around borders
        x_centers, y_centers, z_centers = correct_center_positions(x_centers, y_centers, z_centers, self.tomo.shape, hdim)
        # total number of patches that we should extract
        total_pnum = len(x_centers) * len(y_centers) * len(z_centers)

        # normalize and zero pad the tomogram values
        tomo = (self.tomo - np.mean(self.tomo)) / np.std(self.tomo)
        tomo = np.pad(tomo, 0, mode='constant', constant_values=0)
        # two arrays to collect ther esults of rpediction
        # one for predicted intensity values, second hodls the predicted class labels
        pred_tvals = np.zeros(self.tomo.shape).astype(np.int8)
        pred_tclass = np.zeros(self.tomo.shape + (self.num_class,)).astype(np.float16)  # tomo.shape * # classes

        patch_num = 1
        for z in z_centers:
            for y in y_centers:
                for x in x_centers:
                    display('patch number ' + str(patch_num) + ' out of ' + str(total_pnum))
                    patch = tomo[z - hdim:z + hdim, y - hdim:y + hdim, x - hdim:x + hdim]
                    patch = np.expand_dims(patch, axis=0)  # expanding dimensions for predict function (batch)
                    patch = np.expand_dims(patch, axis=4)  # expanding dimensions for predict function (channel)
                    pred_vals = self.model.predict(patch, batch_size=1)
                    print(np.unique(np.argmax(pred_vals, 4)))

                    # assign predicted values to the corresponding patch location in the tomogram
                    current_patch = pred_tclass[z - hdim:z + hdim, y - hdim:y + hdim, x - hdim:x + hdim, :]
                    casted_pred_vals = np.float16(pred_vals[0, 0:2 * hdim, 0:2 * hdim, 0:2 * hdim, :])
                    pred_tclass[z - hdim:z + hdim, y - hdim:y + hdim,
                    x - hdim:x + hdim] = current_patch + casted_pred_vals

                    # one-hot-encoded for normalization (labels)
                    current_patch2 = pred_tvals[z - hdim:z + hdim, y - hdim:y + hdim, x - hdim:x + hdim]
                    argmax_labels = np.ones((self.patch_size, self.patch_size, self.patch_size), dtype=np.int8)
                    pred_tvals[z - hdim:z + hdim, y - hdim:y + hdim, x - hdim:x + hdim] = current_patch2 + argmax_labels

                    patch_num += 1

        print("Fetching Finished")

        # required only if there are overlapping regions (normalization)
        for n in range(self.num_class):
            pred_tclass[:, :, :, n] = pred_tclass[:, :, :, n] / pred_tvals

        # print(np.unique(np.argmax(pred_tclass, 3)))
        return pred_tclass

    def save_result(self, scoremap_tomo, labelmap_tomo):
        binned_scoremap = self.bin_tomo(scoremap_tomo)
        binned_labelmap = np.int8(np.argmax(binned_scoremap, axis=-1))

        # Save labelmaps:
        labelmap_path = os.path.join(self.output_path, 'tomo_labelmap.mrc')
        binned_labelmap_path = os.path.join(self.output_path, 'tomo_binned_labelmap.mrc')
        write_mrc(labelmap_tomo, labelmap_path)
        write_mrc(binned_labelmap, binned_labelmap_path)
        display_message("Results of labelmap and binned labelmap are saved as mrc files "
                        "in the following paths accordingly :\n"
                        + str(labelmap_path) + "\n"
                        + str(binned_labelmap_path), False)

    def bin_tomo(self, scoremap_tomo):
        from skimage.measure import block_reduce

        bd0 = int(np.ceil(scoremap_tomo.shape[0] / 2))
        bd1 = int(np.ceil(scoremap_tomo.shape[1] / 2))
        bd2 = int(np.ceil(scoremap_tomo.shape[2] / 2))
        new_dim = (bd0, bd1, bd2, self.num_class)
        binned_scoremap = np.zeros(new_dim)

        for cnum in range(self.num_class):
            # averaging over 2x2x2 patches to subsample
            # we divided the dimension by half so each size dimension is subsampled by 2,
            # also apply for each class dimension
            binned_scoremap[:, :, :, cnum] = block_reduce(scoremap_tomo[:, :, :, cnum], (2, 2, 2), np.mean)

        return binned_scoremap

    def start_clustering(self):
        """
        We segment the tomogram to non-overlapped patches
        Args: None (requires just the path to tomogram and the network weights)
        :return: numpy array of scoremap predicted by the trained model
        """
        # inititalize some values
        radi = 5
        thr = 1
        s = 2

        # check the file exists
        labelmap_path = os.path.join(self.output_path, "tomo_binned_labelmap.mrc")
        is_file(labelmap_path)

        binned_labelmap = read_mrc(labelmap_path)
        object_list = self.save_coordinates(binned_labelmap, radi, thr)
        self.scale_coordinates(object_list, (s, s, s))

    def save_coordinates(self, binned_labelmap, radi, thr):
        """
        This function returns coordinates of individual particles. Meanshift clustering is used.
        :param binned_labelmap_tomo: the labelmap to be segmented and analysed
        :param radi: cluster radius
        :param thr: threshold for discarding based on radius
        :return: list of coordinates and corresponding classes of the clustered particles
        """
        # testvals = binned_labelmap[binned_labelmap > 0]
        display("Getting coordinates...")
        obj_vals = np.transpose(np.array(np.nonzero(binned_labelmap > 0)))
        display("Clustering coordinates")
        clusters = MeanShift(bandwidth=radi, bin_seeding=True).fit(obj_vals)
        display("Clustering finished")
        num_clusters = clusters.cluster_centers_.shape[0]

        object_list = []
        labels = np.zeros((self.num_class-1,))
        display("collect data points of clusters...")
        for n in range(num_clusters):
            cluster_data_point_indx = np.nonzero(clusters.labels_ == n)

            # #data points of a cluster and lcoation of centroids:
            cluster_size = np.size(cluster_data_point_indx)
            centroids = clusters.cluster_centers_[n]

            # assign each detected particle to a class of clusters:
            cluster_elements = []
            for c in range(cluster_size):  # get labels of cluster members
                element_coord = obj_vals[cluster_data_point_indx[0][c], :]
                cluster_elements.append(binned_labelmap[element_coord[0], element_coord[1], element_coord[2]])

            for num in range(self.num_class-1):  # get most present label in cluster
                labels[num] = np.size(np.nonzero(np.array(cluster_elements) == num + 1))
            assigned_label = np.argmax(labels) + 1

            object_list = add_obj(object_list, label=assigned_label, coord=centroids, c_size=cluster_size)

        display("Collecting finished")
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

            display('# objects in Class ' + class_names[l] + ': ' + str(len(obj_class_list)))

    def scale_coordinates(self, objlist, scale):
        """
        This function scales the coordinations by a given factor
        Args: objlist a dictionary list of objects generated by save_coordinates function
              scale (tuple of z, y, x): scaling is applied accordingly to each dim
        :return: new dictionary list holding the scaled coordinates
        """
        if not isinstance(scale, tuple):
            display_message("scale must be a (z, y, x) tuple")
            sys.exit()

        # avoid overwriting the set of components copied.
        new_objlist = deepcopy(objlist)
        for i in range(len(objlist)):
            x = int(np.round(float(objlist[i]['x'])))
            y = int(np.round(float(objlist[i]['y'])))
            z = int(np.round(float(objlist[i]['z'])))
            new_objlist[i]['x'] = scale[2] * x
            new_objlist[i]['y'] = scale[1] * y
            new_objlist[i]['z'] = scale[0] * z
            new_objlist[i]['obj_id'] = i
            new_objlist[i]['tomo_idx'] = 9

        # drop small particles based on threshold for each class
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        thresholds = [50, 100, 20, 100, 50, 100, 100, 50, 50, 20, 300, 300]

        clean_objlist = []
        for l in labels:
            # get the class of coords
            idx_class = []
            for idx in range(len(new_objlist)):
                if str(new_objlist[idx]['label']) == str(l):
                    idx_class.append(idx)

            objlist_class = []
            for idx in range(len(idx_class)):
                objlist_class.append(new_objlist[idx_class[idx]])

            # filter out
            thr_indices = []
            for idx in range(len(objlist_class)):
                cluster_size = objlist_class[idx]['c_size']
                if cluster_size is not None:
                    if cluster_size >= thresholds[l - 1]:
                        thr_indices.append(idx)
                else:
                    print('no attribute cluster size (c_size) is defined for object ' + str(idx))

            objlist_filtered = []
            for idx in range(len(thr_indices)):
                objlist_filtered.append(objlist_class[thr_indices[idx]])

            for p in range(len(objlist_class)):
                clean_objlist.append(objlist_class[p])

        raw_ploc_path = os.path.join(self.output_path, 'tomo_objlist_raw.xml')
        scaled_ploc_path = os.path.join(self.output_path, 'tomo_objlist_thr.xml')

        write_xml(objlist, raw_ploc_path)
        write_xml(clean_objlist, scaled_ploc_path)

        display_message("The raw and scaled versions of the coordinates are saved as xml files "
                        "in the following paths accordingly: \n "
                        + str(scaled_ploc_path) + " \n"
                        + str(raw_ploc_path), False)

    def start_evaluation(self):
        list_classes = class_names
        clean_objlist_path = os.path.join(self.output_path, 'tomo_objlist_thr.xml')
        clean_objlist = read_xml2(clean_objlist_path)
        particle_lcoations = os.path.join(self.output_path, 'particle_locations_tomo9.txt')
        file = open(particle_lcoations, 'w')
        for p in range(len(clean_objlist)):
            x = int(clean_objlist[p]['x'])
            y = int(clean_objlist[p]['y'])
            z = int(clean_objlist[p]['z'])
            l = int(clean_objlist[p]['label'])
            file.write(list_classes[l] + ' ' + str(x) + ' ' + str(y) + ' ' + str(z) + '\n')
        file.close()
        display_message("Particle lcoations are saved as a txt file in " + str(particle_lcoations), False)
        print("generating final results...")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    set_theme_style(app)

    application = EvaluationWindow()
    application.show()

    sys.exit(app.exec_())
