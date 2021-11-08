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
from scipy.spatial import distance
from contextlib import redirect_stdout

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QIcon
from gui.evaluation import evaluation
from gui.theme_style import *
from utils.params import *
from train_models import *
from sklearn.cluster import MeanShift, KMeans
from sklearn.metrics import confusion_matrix
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
        self.model_names = ["3D UNet"]  # "YOLOv3", "R-CNN", "Mask R-CNN"]
        self.generate_model_radio_btns(1)

        self.num_class = 13
        self.patch_size = 160
        self.patch_crop = 20
        self.patch_overlap = 55
        self.slide = None
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
        print("Segmentation Finished")
        sys.exit()

    def extract_patches(self):
        display("fetching patches...")
        # self.patch_size = 160
        # self.patch_crop = 25
        # self.patch_overlap = 55
        bwidth = int(self.patch_size / 2)
        bcrop = int(bwidth - self.patch_crop)
        slide = int(2 * bwidth + 1 - self.patch_overlap)  # self.patch_size + 1
        # normalize and zero pad the tomogram values
        tomo = (self.tomo - np.mean(self.tomo)) / np.std(self.tomo)
        tomo = np.pad(tomo, self.patch_crop, mode='constant', constant_values=0)
        # self.tomo = tomo

        # z_half_dim, y_half_dim, x_half_dim = (self.patch_size / 2, self.patch_size / 2, self.patch_size / 2)
        x_centers = list(range(bwidth, tomo.shape[2] - bwidth, slide))
        y_centers = list(range(bwidth, tomo.shape[1] - bwidth, slide))
        z_centers = list(range(bwidth, tomo.shape[0] - bwidth, slide))

        # if dimensions are not exactly divisible,
        # we should collect the remained voxels around borders
        x_centers, y_centers, z_centers = correct_center_positions(x_centers, y_centers, z_centers, tomo.shape, bwidth)
        # total number of patches that we should extract
        total_pnum = len(x_centers) * len(y_centers) * len(z_centers)

        # two arrays to collect ther esults of rpediction
        # one for predicted intensity values, second hodls the predicted class labels
        pred_tvals = np.zeros(tomo.shape).astype(np.int8)
        pred_tclass = np.zeros(tomo.shape + (self.num_class,)).astype(np.float16)  # tomo.shape * # classes

        patch_num = 1
        for z in z_centers:
            for y in y_centers:
                for x in x_centers:
                    display('patch number ' + str(patch_num) + ' out of ' + str(total_pnum))
                    patch = tomo[z-bwidth:z+bwidth, y-bwidth:y+bwidth, x-bwidth:x+bwidth]
                    patch = np.expand_dims(patch, axis=0)  # expanding dimensions for predict function (batch)
                    patch = np.expand_dims(patch, axis=4)  # expanding dimensions for predict function (channel)
                    pred_vals = self.model.predict(patch, batch_size=1)
                    print(np.unique(np.argmax(pred_vals, 4)))

                    # assign predicted values to the corresponding patch location in the tomogram
                    current_patch = pred_tclass[z-bcrop:z+bcrop, y-bcrop:y+bcrop, x-bcrop:x+bcrop, :]
                    lowb = bwidth - bcrop
                    highb = bwidth + bcrop
                    casted_pred_vals = np.float16(pred_vals[0, lowb:highb, lowb:highb, lowb:highb, :])
                    pred_tclass[z-bcrop:z+bcrop, y-bcrop:y+bcrop, x-bcrop:x+bcrop] = current_patch + casted_pred_vals

                    # one-hot-encoded for normalization (labels)
                    current_patch2 = pred_tvals[z-bcrop:z+bcrop, y-bcrop:y+bcrop, x-bcrop:x+bcrop]
                    size_dim = self.patch_size - 2 * self.patch_crop
                    argmax_labels = np.ones((size_dim, size_dim, size_dim), dtype=np.int8)
                    pred_tvals[z-bcrop:z+bcrop, y-bcrop:y+bcrop, x-bcrop:x+bcrop] = current_patch2 + argmax_labels

                    patch_num += 1

        print("Fetching Finished")

        # required only if there are overlapping regions (normalization)
        for n in range(0, self.num_class):
            pred_tclass[:, :, :, n] = pred_tclass[:, :, :, n] / pred_tvals

        # write_mrc(preds, os.path.join(self.output_path, "probabilities.mrc"))
        pred_tclass = pred_tclass[self.patch_crop:-self.patch_crop, self.patch_crop:-self.patch_crop, self.patch_crop:-self.patch_crop, :]  # unpad
        print(np.unique(np.argmax(pred_tclass, 3)))
        return pred_tclass

    def save_result(self, scoremap_tomo, labelmap_tomo):
        binned_scoremap = self.bin_tomo(scoremap_tomo)
        binned_labelmap = np.int8(np.argmax(binned_scoremap, axis=-1))

        # Save labelmaps:
        scoremap_path = os.path.join(self.output_path, 'segment/scoremap_tomo.mrc')
        labelmap_path = os.path.join(self.output_path, 'segment/tomo_labelmap.mrc')
        binned_labelmap_path = os.path.join(self.output_path, 'segment/tomo_binned_labelmap.mrc')
        write_mrc(scoremap_tomo, scoremap_path)
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
        labelmap_path = os.path.join(self.output_path, "segment/tomo_binned_labelmap.mrc")
        is_file(labelmap_path)

        binned_labelmap = read_mrc(labelmap_path)
        object_list = self.save_coordinates(binned_labelmap, radi, thr)
        self.scale_coordinates(object_list, (s, s, s))
        print("Clustering Finished")
        sys.exit()

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
        # clusters = KMeans(n_clusters=self.num_class, random_state=0).fit(obj_vals)
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

        raw_ploc_path = os.path.join(self.output_path, 'cluster/tomo_objlist_raw.xml')
        scaled_ploc_path = os.path.join(self.output_path, 'cluster/tomo_objlist_thr.xml')

        write_xml(objlist, raw_ploc_path)
        write_xml(clean_objlist, scaled_ploc_path)

        display_message("The raw and scaled versions of the coordinates are saved as xml files "
                        "in the following paths accordingly: \n "
                        + str(scaled_ploc_path) + " \n"
                        + str(raw_ploc_path), False)

    def start_evaluation(self):
        # initialziation
        list_classes = class_names
        gtcoord_flag = True
        gt_z_offset = 156
        gt_z_offset = gtcoord_flag * gt_z_offset
        padding = 0
        total_num_hit = 0
        total_dist = 0
        clipped_ptls = 0

        # set paths
        clean_objlist_path = os.path.join(self.output_path, 'cluster/tomo_objlist_thr.xml')
        clean_objlist = read_xml2(clean_objlist_path)
        gt_ptls_path = os.path.join(self.output_path, "particle_locations_model.txt")
        res_ptls_path = os.path.join(self.output_path, 'particle_locations_tomo.txt')
        hitbox_path = os.path.join(self.output_path, "hitbox.mrc")

        is_file(clean_objlist_path)
        is_file(hitbox_path)
        is_file(gt_ptls_path)
        is_file(hitbox_path)

        display("Evaluation started")
        # read the particle locations
        file = open(res_ptls_path, 'w')
        for p in range(len(clean_objlist)):
            x = int(clean_objlist[p]['x'])
            y = int(clean_objlist[p]['y'])
            z = int(clean_objlist[p]['z'])
            l = int(clean_objlist[p]['label'])
            file.write(list_classes[l] + ' ' + str(x) + ' ' + str(y) + ' ' + str(z) + '\n')
        file.close()
        # display("Particle lcoations are saved as a txt file in " + str(res_ptls_path))
        # display("Generating final results...")

        is_file(res_ptls_path)

        # loading results
        pred_ptls = []
        with open(str(res_ptls_path), 'rU') as f:
            for line in f:
                pcn, zcoord, ycoord, xcoord, *_ = line.rstrip('\n').split()
                x = round(float(zcoord))
                y = round(float(ycoord))
                z = round(float(xcoord))
                pred_ptls.append((pcn, int(z), int(y), int(x)))

        # loading ground truth
        gt_ptls = []
        with open(gt_ptls_path, 'rU') as f:
            for line in f:
                pcn_idx, zcoord, ycoord, xcoord, rot_1, rot_2, rot_3 = line.rstrip('\n').split()
                x = round(float(zcoord))
                y = round(float(ycoord))
                z = round(float(xcoord))
                gt_ptls.append((pcn_idx, int(z), int(y), int(x)))
        gt_ptls_cls = np.asarray([reversed_class_names[p[0]] for p in gt_ptls])
        hitbox = read_mrc(hitbox_path)

        # variable definition
        pred_ptls_num_hits = np.zeros((len(gt_ptls),), dtype=int)
        pred_ptls_num_hitcls = {key: 0 for key, val in reversed_class_names.items()}
        pred_ptls_cls = np.zeros_like(gt_ptls_cls)

        predicted_particles = [(p[0], p[1] - gt_z_offset - padding, p[2] - padding, p[3] - padding) for p in pred_ptls]

        # valdiate each particle
        for idx, pt in enumerate(pred_ptls):
            # force coordinates to be within the indices borders of the tomogram (clipping)
            pcn, z, y, x = pt
            zcoord = max(min(z, hitbox.shape[0] - 1), 0)
            ycoord = max(min(y, hitbox.shape[1] - 1), 0)
            xcoord = max(min(x, hitbox.shape[2] - 1), 0)

            # count number of clipped particles
            if zcoord != z or ycoord != y or xcoord != x:
                clipped_ptls += 1

            # get the corresponding id of the particle located at (z, y, x)
            ptl = int(hitbox[z][y][x])

            # check if there is a particle at that location (missed)
            if ptl == 0:
                pred_ptls_num_hitcls['bg'] += 1
                continue

            # we consider indexing particles from 0, but the given hitbox indexed particles from 1 (0 does not exist)
            ptl -= 1
            pred_ptls_num_hits[ptl] += 1
            pred_ptls_num_hitcls[pcn] += 1  # add one unit to the num of particles detected for corresponding class pt
            total_num_hit += 1  # keep counting of total number of hits (over all classes)

            # find ground truth center
            real_particle = gt_ptls[ptl]  # ground truth particle
            true_center = (real_particle[1], real_particle[2], real_particle[3])

            # compute euclidean distance from predicted center to real center
            total_dist += np.abs(distance.euclidean((z, y, x), true_center))

            # use only the first classification prediction for that particle
            if pred_ptls_cls[ptl] == 0:
                pred_ptls_cls[ptl] = reversed_class_names[pcn]

        # report some numbers
        unique_particles_found = sum([int(p >= 1) for p in pred_ptls_num_hits])
        unique_particles_not_found = sum([int(p == 0) for p in pred_ptls_num_hits])
        multiple_hits = sum([int(p > 1) for p in pred_ptls_num_hits])

        total_recall = unique_particles_found / len(gt_ptls_cls)
        total_precision = unique_particles_found / len(predicted_particles)
        total_f1 = 1 / ((1 / total_recall + 1 / total_precision) / 2)
        total_missrate = unique_particles_not_found / len(gt_ptls)
        avg_distance = total_dist / total_num_hit

        cm = ConfusionMatrix(actual_vector=gt_ptls_cls, predict_vector=pred_ptls_cls)
        # relabel confusion matrix
        class_labels = class_names.copy()
        # check whether all classes are represented and if not, remove them from label dict
        # for k in class_names:
        #     if k not in cm.classes:
        #         class_labels.pop(k)
        # cm.relabel(class_labels)

        # save results as a log file
        log_path = os.path.join(self.output_path, 'evaluate/evaluation_log.txt')
        with open(log_path, 'w') as f:
            with redirect_stdout(f):
                print('EVALUATION results for localization')
                print(f'Found {len(predicted_particles)} results')
                print(
                    f'TP: {unique_particles_found} unique particles localized out of total {len(gt_ptls)} particles')
                print(f'FP: {pred_ptls_num_hitcls["bg"]} reported particles are false positives')
                print(f'FN: {unique_particles_not_found} unique particles not found')
                if multiple_hits:
                    print(f'Note: there were {multiple_hits} unique particles that had more than one result')
                if clipped_ptls:
                    print(f'Note: there were {clipped_ptls} results that were outside of tomo bounds ({hitbox.shape})')
                print(f'Average euclidean distance from predicted center to ground truth center: {avg_distance:.5f}')
                print(f'Total recall: {total_recall:.5f}')
                print(f'Total precision: {total_precision:.5f}')
                print(f'Total miss rate: {total_missrate:.5f}')
                print(f'Total f1-score: {total_f1:.5f}')
                print('\nEVALUATION results for classification')
                print(cm)

        cm.save_html(os.path.join(self.output_path, 'evaluate/classification_log'))
        cm.save_csv(os.path.join(self.output_path, 'evaluate/classification_log'))
        # save confusion matrix as a plot
        cnf_matrix = confusion_matrix(gt_ptls_cls, pred_ptls_cls)
        cnf_matrix = cnf_matrix[1:cnf_matrix.shape[0], 1:cnf_matrix.shape[1]]  # remove background class
        class_lbls = np.transpose(list(class_names.values()))
        class_lbls = class_lbls[1:len(class_lbls)]
        # # plot normalized and non-normalized confusion matrix
        # plt.figure(num=1, figsize=(10, 10), dpi=150)
        # plot_confusion_matrix(cnf_matrix, classes=class_lbls, eps_dir=self.output_path)
        #
        # plt.figure(num=2, figsize=(10, 10), dpi=150)
        plot_confusion_matrix(cnf_matrix, classes=class_lbls, eps_dir=self.output_path, normalize=True)
        plot_confusion_matrix(cnf_matrix, classes=class_lbls, eps_dir=self.output_path, normalize=False)

        # Plot all ROC curves
        # ROC curves are appropriate when the observations are balanced between each class ,
        # whereas precision-recall curves are appropriate for imbalanced datasets.

        # scoremap_path = os.path.join(self.output_path, 'segment/scoremap_tomo.mrc')
        # score_tomo = read_mrc(scoremap_path)
        # mask_tomo = read_mrc(os.path.join(self.output_path, 'target_grandmodel_9.mrc'))
        # mask_onehot = to_categorical(mask_tomo, self.num_class)
        # # plot roc for this patch
        # mask_onehot = mask_onehot.reshape((mask_onehot.shape[0]*mask_onehot.shape[1]*mask_onehot.shape[2]), 13)
        # score_tomo = score_tomo.reshape((score_tomo.shape[0] * score_tomo.shape[1] *
        #                                    score_tomo.shape[2]), 13)
        # plt.figure(num=3, figsize=(8, 6), dpi=80)
        # plot_roc(mask_onehot, score_tomo, self.num_class, self.output_path)
        # plt.figure(num=4, figsize=(8, 6), dpi=80)
        # plot_recall_precision(mask_onehot, score_tomo, self.num_class, self.output_path)
        plt.show()

        display("Evaluation finished")
        sys.exit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    set_theme_style(app)

    application = EvaluationWindow()
    application.show()

    sys.exit(app.exec_())
