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
import re
import time
import math
import shutil
# import random
import losses
from skimage.util import random_noise
import string
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
# from sklearn.model_selection import train_test_split
# from tensorflow.python.keras import optimizers
# import numpy as np
# import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, r2_score
from sklearn.preprocessing import MinMaxScaler
# from utils.cyclicR import CyclicLR
# from utils.params import *
import params
from utils.plots import *
from utils.utility_tools import *
# from utils.CyclicLR.clr_callback import CyclicLR
from utils.losses import *
from utils.models import *

from keras.losses import Huber
from keras.optimizers import *
from keras import metrics
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau  # LearningRateScheduler
import keras.backend as k
from glob import glob
# import elasticdeform
# import torchio as tio
# from sklearn.utils import class_weight
# import tensorflow_probability as tfp
import tensorflow as tf

class TrainModel:

    def __init__(self, obj):
        # define values
        self.net = None
        self.optimizer = None  # "Adam"
        self.process_time = None
        self.class_weights = None
        self.list_tomos_IDs = None
        self.list_masks_IDs = None
        self.list_annotations = None

        self.train_acc = 0
        self.vald_acc = 0
        self.train_loss = 0
        self.vald_loss = 0
        self.f1_score = 0
        self.precision = 0
        self.recall = 0
        self.pathes_test = 0
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.base_lr = 0.01
        self.max_lr = 0.1
        self.step_size = 1
        self.scale_fn = None
        self.scale_mode = 'cycle'

        # values to collect outputs
        self.lr_history = {}
        self.model_history = []
        self.history_lr = []
        self.history_recall = []
        self.history_vald_acc = []
        self.history_f1_score = []
        self.history_train_acc = []
        self.history_vald_loss = []
        self.history_precision = []
        self.history_train_loss = []
        self.patches_tomos = []
        self.patches_masks = []

        # initialize values
        self.obj = obj
        print(self.obj.loss)
        self.train_img_path = os.path.join(obj.base_path, "images/")
        self.train_target_path = os.path.join(obj.base_path, "targets/")
        self.weight_path = os.path.join(obj.base_path, obj.weight_path)
        self.lr = obj.lr
        self.lr_type = obj.lr_type  # "exp_decay"
        self.initial_lr = obj.lr
        self.width = obj.img_dim[0]
        self.height = obj.img_dim[1]
        if obj.dim_num > 2:
            self.depth = obj.img_dim[2]

        # create output directory
        seednum = 2
        np.random.seed(seednum)
        random_str = ''.join([random.choice(string.ascii_uppercase + string.digits) for n in range(16)])
        self.output_path = os.path.join(obj.output_path, random_str)
        os.makedirs(self.output_path)

        display("The output path is: ")
        display(self.output_path)

        # check values
        is_positive(self.obj.epochs, 'epochs')
        is_positive(self.obj.classNum, 'num_class')
        is_dir(self.train_img_path)
        is_dir(self.train_target_path)
        is_dir(self.output_path)

        # load the whole dataset
        self.fetch_dataset()

        # initialize model
        self.train_model()

    def fetch_dataset(self):
        """ this functions generates a list of file names from
        all available tomograms in train_img_path fodler
        and their corresponding masks in train_target_path folder
        """
        display("fetching dataset...")
        self.list_tomos_IDs = glob(os.path.join(self.train_img_path, "*.mrc"))
        self.list_masks_IDs = glob(os.path.join(self.train_target_path, "*.mrc"))

        # check if every tomo has a corresponding mask
        if len(self.list_tomos_IDs) != len(self.list_masks_IDs):
            display("Expected two" + str(len(self.list_tomos_IDs)) + ", received " +
                    str(len(self.list_tomos_IDs)) + " and " + str(len(self.list_masks_IDs)) +
                    ". \n There is a missing pair of tomogram and target.")
            sys.exit()

        self.list_tomos_IDs.sort(key=lambda f: int(re.sub('\D', '', f)))
        self.list_masks_IDs.sort(key=lambda r: int(re.sub('\D', '', r)))
        # print(self.list_tomos_IDs)
        # self.fetch_tomos()

    def fetch_tomos(self, tomo_indices):
        """
        this function reads mrc files of tomograms and their corresponding masks from the list of files generated by
        fetch_dataset() function.
        Then it generates patches of size patch_size * patch_size * patch_size
        :return patches_tomos: patches generated from all tomograms in order
                patches_masks: patches generated from all masks in order
        """
        # display("fetched tomograms " +  tomo_indices)
        # print("Unique IDs:", tomo_indices)
        start = time.perf_counter()
        # for t in range(len(self.list_tomos_IDs)):
        self.patches_tomos = []
        self.patches_masks = []
        for t in tomo_indices:
            # display("********** Fetch Tomo {tnum} **********".format(tnum=self.list_tomos_IDs[t]))
            tomo = read_mrc(self.list_tomos_IDs[t])
            tomo = np.flip(tomo, axis=1)
            mask = read_mrc(self.list_masks_IDs[t])
            mask = np.flip(mask, axis=1)

            # check if the tomogram and its mask are of the same size
            if tomo.shape != mask.shape:
                display("the tomogram and the target must be of the same size. " +
                        str(tomo.shape) + " is not equal to " + str(mask.shape) + ".")
                sys.exit()

            self.patches_tomos.append(tomo)
            self.patches_masks.append(mask)



        # Alternative data loading approach as a pytorch tensor, ready to use for GPU
        # preparation of tomogram as a tensor that can be used with tensorflow API
        # self.tomo = np.swapaxes(self.tomo, 0, 2)  # changing dimension order from (z, y, x) to (x, y, z)
        # self.tomo = np.expand_dims(self.tomo, axis=0)  # expanding dimensions for tensorflow input
        # self.tomo = np.expand_dims(self.tomo, axis=4)  # expanding dimensions for tensorflow input
        #
        # # preparation of mask as a tensor that can be used with tensorflow API
        # self.mask = np.swapaxes(self.mask, 0, 2)  # changing dimension order from (z, y, x) to (x, y, z)
        # self.mask = np.expand_dims(self.mask, axis=0)  # expanding dimensions for tensorflow input
        # self.mask = np.expand_dims(self.mask, axis=4)  # expanding dimensions for tensorflow input
        #
        # # extracting patches of size patch_size * patch_size * patch_size
        # patch_tomo = tf.extract_volume_patches(self.tomo,
        #                                        [1, self.patch_size, self.patch_size, self.patch_size, 1],
        #                                        [1, self.patch_size, self.patch_size, self.patch_size, 1],
        #                                        padding='VALID')
        # patch_tomo = tf.reshape(patch_tomo, [-1, self.patch_size, self.patch_size, self.patch_size])
        # patch_tomo = tf.squeeze(patch_tomo)
        #
        # # extracting patches of size patch_size * patch_size * patch_size
        # patch_mask = tf.extract_volume_patches(self.mask,
        #                                        [1, self.patch_size, self.patch_size, self.patch_size, 1],
        #                                        [1, self.patch_size, self.patch_size, self.patch_size, 1],
        #                                        padding='VALID')
        # patch_mask = tf.reshape(patch_mask, [-1, self.patch_size, self.patch_size, self.patch_size])
        # patch_mask = tf.squeeze(patch_mask)
        #
        # # converting back from tensor to numpy
        # patch_tomo = patch_tomo.eval(session=tf.compat.v1.Session())
        # patch_mask = patch_mask.eval(session=tf.compat.v1.Session())
        #
        # # the images are
        # patch_tomo = np.swapaxes(patch_tomo, 1, 3)  # changing back dimension order from (x, y, z) to (z, y, x)
        # patch_mask = np.swapaxes(patch_mask, 1, 3)  # changing back dimension order from (x, y, z) to (z, y, x)
        #
        # # concatenating all patches into a signle array (in order)
        # if t == 0:
        #     self.patches_tomos = patch_tomo
        #     self.patches_masks = patch_mask
        # else:
        #     self.patches_tomos = np.concatenate((self.patches_tomos, patch_tomo))
        #     self.patches_masks = np.concatenate((self.patches_masks, patch_mask))
        # display("********** END Fetch Tomo {tnum} **********".format(tnum=self.list_tomos_IDs[t]))

        end = time.perf_counter()
        process_time = (end - start)
        # display("tomograms and annotations fetched in {:.2f} seconds.".format(round(process_time, 2)))

    def train_model(self):
        """This function starts the training procedure by calling
           different built-in functions of the class CNNModel
        """
        self.get_model()  # build the CNN model
        self.fit_model()  # fit the data to the model and train the model
        self.plots()  # plot the results
        self.save()  # save the results as txt

        plt.show(block=True)

    def get_model(self):
        cnnobj = CNNModels()
        cnnobj.cls_reg = self.obj.train_type
        # self.obj.model_type = "classification"
        if self.obj.model_type == "2D UNet":
            self.net = cnnobj.unet2d((self.obj.patch_size, self.obj.patch_size), self.obj.classNum)
        elif self.obj.model_type == "3D UNet":
            self.net = cnnobj.unet3d((self.obj.patch_size, self.obj.patch_size, self.obj.patch_size), self.obj.classNum)
        elif self.obj.model_type == "TL 3D UNet":
            # pretrained_net = cnnobj.unet3d_encoder((self.obj.patch_size, self.obj.patch_size, self.obj.patch_size))
            # pretrained_weight_path = '/media/noushin/Data/Cryo-ET/DeepET/data2/results/pretraining/real_vs_synthetic/real_synthetic_pretraining_weights.h5'
            # pretrained_net.load_weights(pretrained_weight_path)
            self.net = cnnobj.unet3d((self.obj.patch_size, self.obj.patch_size, self.obj.patch_size), self.obj.classNum)
            self.net.load_weights(self.weight_path)

            # freezing specific layers
            # i = 1
            # for layer in pretrained_net.layers[1:11]:
            #     j = 0
            #     for layer2 in pretrained_net.layers:
            #         if layer.name == layer2.name:
            #             try:
            #                 print(pretrained_net.layers[i].name)
            #                 print(self.net.layers[j].name)
            #                 realvssynt_weights = pretrained_net.layers[i].weights[0]
            #                 TL_weights = self.net.layers[j].weights[0]
            #                 avg_weights = (TL_weights + realvssynt_weights) / 2
            #                 self.net.layers[j].set_weights([avg_weights, self.net.layers[j].bias])
            #             except:
            #                 pass
            #         j += 1
            #     i += 1

            #     # freezing specific layers
            layer_freezing_list = ['conv3d_4', 'conv3d_5', 'conv3d_6', 'conv3d_7']
            for layer in self.net.layers:
                if layer.name in layer_freezing_list:
                    layer.trainable = False
            print("Started Transfer Learning by loading weights...")
            #     ############## Code we used so far for TL#########

            # print("Started Transfer Learning by loading weights...")

        # elif self.obj.model_type == "TL 3D UNet":
        #     self.net = cnnobj.unet3d((self.obj.patch_size, self.obj.patch_size, self.obj.patch_size), self.obj.classNum)
        #
        #     ############## Code we used so far for TL#########
        #     self.net = cnnobj.unet3d((self.obj.patch_size, self.obj.patch_size, self.obj.patch_size), self.obj.classNum)
        #     self.net.load_weights(self.weight_path)
        #
        #     # freezing specific layers
        #     layer_freezing_list = ['conv3d_4', 'conv3d_5', 'conv3d_6', 'conv3d_7']
        #     for layer in self.net.layers:
        #         if layer.name in layer_freezing_list:
        #             layer.trainable = False
        #     print("Started Transfer Learning by loading weights...")
        #     ############## Code we used so far for TL#########




        #     to train only the decoder
        #     for layer in self.net.layers[:11]:
        #         layer.trainable = False
        #
        #     Create the model again
        #     from keras import models
        #     model = models.Sequential()
        #     # Add the 3d-UNet base model but this time without the classification layer
        #     model.add(self.net)
        #     # Add a new classification layer
        #     model.add(layers.Conv3D(len(self.obj.class_names), (1, 1, 1),
        #                             padding='same', activation='softmax', name="cls_layer"))
        # elif self.obj.model_type == "classification":
        #     self.net = cnnobj.cnn3d((self.obj.patch_size, self.obj.patch_size, self.obj.patch_size))

        print("NUM GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        print(self.net.summary())  # TF_ENABLE_ONEDNN_OPTS=0
        # print(cnnobj.get_model_memory_usage(24, self.net))

        # set the properties of the mdoel
        self.set_optimizer()
        self.set_compile()

    def fit_model(self):
        label_list = []
        for l_list in range(self.obj.classNum):
            label_list.append(l_list)
        start = time.perf_counter()


        # if you use size of generated tensor it would be more accurate and it will never throw error
        # len(int(np.round(len(.9 * self.list_annotations)))/self.obj.batch_size) - 1
        # self.selected_tomo_indices = []
        self.list_annotations = read_xml2(os.path.join(self.train_img_path, "object_list_train.xml"))
        # num_tomos = list(range(0, (len(self.list_tomos_IDs))))
        # random.shuffle(num_tomos)
        # teils = int(len(self.list_tomos_IDs) / 3)
        # for tindx in range(0, teils):
        #     self.selected_tomo_indices.append(num_tomos[tindx: tindx + 3])
        # selected_tomos = self.list_tomos_IDs[selected_tomo_indx]
        # self.list_annotations = total_list_annotations["tomo_id" == selected_tomo_indx]
        steps_per_epoch = int(np.round((1-self.obj.vald_prc) * len(self.list_annotations)) / self.obj.batch_size)
        vald_steps_per_epoch = int(np.round(self.obj.vald_prc * len(self.list_annotations)) / self.obj.batch_size) - 1
        counter = 0

        print("Train - Steps per epoch: ", steps_per_epoch)
        print("Validation - Steps per epoch: ", vald_steps_per_epoch)

        # print(tf.config.list_physical_devices('GPU'))
        # print(tf.test.is_gpu_available(cuda_only=True) )
        for e in range(self.obj.epochs):
            flag_new_epoch = True
            print("########## New Epoch ##########\n")
            self.lr = self.initial_lr
            list_train_loss = []
            list_train_acc = []
            list_vald_acc = []
            list_vald_loss = []
            list_f1_score = []
            list_recall = []
            list_precision = []
            list_lr = []
            # self.lr_type = "cyclic"
            self.set_lr(counter)
            if self.lr_type == "Cyclic":
                self.on_train_begin()

            for b in range(steps_per_epoch):
                # fetch the current batch of patches
                if b != 0:
                    flag_new_epoch = False

                # batch_tomo, batch_mask = self.fetch_batch(e, bsize=self.obj.batch_size)
                # Split the data to train and validation (it shuffles the data so the order of patches is not the same )
                # x_train, x_vald, y_train, y_vald = train_test_split(batch_tomo, batch_mask,
                #                                                     test_size=0.2, shuffle=False)
                # display("train size: {x_trainsize}".format(x_trainsize=x_train.shape))
                # y_train = to_categorical(y_train, self.obj.classNum)
                # y_vald = to_categorical(y_vald, self.obj.classNum)

                # expanding dimensions to become suitable for the model input
                # x_train = np.expand_dims(x_train, axis=4)
                # x_vald = np.expand_dims(x_vald, axis=4)
                # y_train = np.array(y_train)
                # y_vald = np.array(y_vald)

                # with LR decay

                # self.lr_type = "cyclic"
                # current_learning_rate = self.set_lr(counter)
                # k.set_value(self.net.optimizer.lr, current_learning_rate)
                list_lr.append(k.eval(self.net.optimizer.lr))
                batch_tomo, batch_mask = self.fetch_batch(b, e, self.obj.batch_size, flag_new_epoch, "Train")

                # collecting current LR
                # current_learning_rate = k.eval(self.net.optimizer.lr)
                # list_lr.append(current_learning_rate)

                # for layer in self.net.layers:
                #     if 'conv' not in layer.name:
                #         continue
                #     self.save_layer_output(batch_tomo, layer_name=layer.name)
                #     filters, biases = layer.get_weights()


                list_layers = ["conv3d_3", "conv3d_13", "reg_layer"]
                if b == 0 and e == 0:
                    for layer in self.net.layers:
                        if layer.name in list_layers:
                            self.save_layer_output(batch_tomo, layer_name=layer.name)
                            self.save_layer_filter(layer, layer_name=layer.name)
                    # print(batch_tomo.shape)
                # for layer in self.net.layers:
                #     print(layer.trainable)
                # self.class_weights = class_weight.compute_class_weight(class_weight='balanced',
                #                                                        classes=np.array([0, 1]),
                #                                                        y=batch_mask.ravel())
                if self.obj.train_type == "segmentation":
                    loss_train = self.net.train_on_batch(batch_tomo, batch_mask, class_weight=self.class_weights)
                    display('epoch %d/%d - b %d/%d - loss: %.3f - acc: %.3f - lr: %.7f' % (e + 1, self.obj.epochs,
                                                                                           b + 1, steps_per_epoch,
                                                                                           loss_train[0],
                                                                                           loss_train[1] * 0.1,
                                                                                           k.eval(
                                                                                               self.net.optimizer.lr)))
                else:
                    loss_train = self.net.train_on_batch(batch_tomo, batch_mask)
                    display('epoch %d/%d - b %d/%d - MSE: %.3f - MAE: %.3f - lr: %.7f' % (
                        e + 1, self.obj.epochs,
                        b + 1, steps_per_epoch,
                        loss_train[0],
                        loss_train[1],
                        k.eval(self.net.optimizer.lr)))

                list_train_loss.append(loss_train[0])
                list_train_acc.append(loss_train[1])
                if self.lr_type == "Cyclic":
                    self.on_batch_end()
                counter = counter + 1

            for d in range(vald_steps_per_epoch):
                batch_tomo_vald, batch_mask_vald = self.fetch_batch(d, e, self.obj.batch_size, False, "Validation")

                # evaluate trained model on the validation set
                loss_val = self.net.evaluate(batch_tomo_vald, batch_mask_vald, verbose=0)
                batch_pred = self.net.predict(batch_tomo_vald)
                # write_mrc2(batch_tomo_vald[0, :, :, :, 0], os.path.join('/media/noushin/Data/Cryo-ET/DeepET/data2/temp/patch_tomo_vald.mrc'))
                # write_mrc2(np.average(batch_pred[0, :, :, :, :], axis=3), os.path.join('/media/noushin/Data/Cryo-ET/DeepET/data2/temp/patch_mask_vald.mrc'))
                if self.obj.train_type == "segmentation":
                    scores = precision_recall_fscore_support(batch_mask_vald.argmax(axis=-1).flatten(),
                                                             batch_pred.argmax(axis=-1).flatten(), average=None,
                                                             labels=label_list, zero_division=0)
                    print("val. loss: {vl}, \n val acc: {va}".format(vl=np.round(loss_val[0], 2),
                                                                     va=np.round(loss_val[1], 2)))
                    print("F1 Score : {f1s}, \n Recall: {res}, \n Precision: {prs}".format(f1s=np.round(scores[2], 2),
                                                                                           res=np.round(scores[1], 2),
                                                                                           prs=np.round(scores[0], 2)))
                    list_f1_score.append(scores[2])
                    list_recall.append(scores[1])
                    list_precision.append(scores[0])
                    print(np.unique(np.argmax(batch_pred, 4)))
                else:
                    # scores = r2_score(batch_mask_vald.argmax(axis=-1).flatten(),
                    #          batch_pred.argmax(axis=-1).flatten(), multioutput='variance_weighted')
                    scores = r2_score(batch_mask_vald.flatten(),
                                      batch_pred.flatten(), multioutput='variance_weighted')
                    # scores = coefficient_of_determination(batch_mask_vald.argmax(axis=-1).flatten(),
                    #                                       batch_pred.argmax(axis=-1).flatten())
                    print("MSE: {vl}, MAE: {va}, R2: {r2}".format(vl=np.round(loss_val[0], 2),
                                                                  va=np.round(loss_val[1], 2),
                                                                  r2=np.round(scores, 2)))
                    list_recall.append(scores)
                    # list_precision.append(loss_val[0])

                list_vald_loss.append(loss_val[0])
                list_vald_acc.append(loss_val[1])

            self.history_train_loss.append(list_train_loss)
            self.history_lr.append(list_lr)
            self.history_train_acc.append(list_train_acc)
            self.history_vald_loss.append(list_vald_loss)
            self.history_vald_acc.append(list_vald_acc)
            if self.obj.train_type == "segmentation":
                self.history_f1_score.append(list_f1_score)
                self.history_recall.append(list_recall)
                self.history_precision.append(list_precision)

            print("################################################################################\n")
            if (e + 1) % 50 == 0:
                self.set_weight_callback(e + 1)

        self.save_history(batch_tomo)
        self.net.save(self.output_path + '/model_final_weights.h5')
        end = time.perf_counter()
        self.process_time = (end - start)
        display(self.process_time)

    def get_balance_data(self, annotated_list, bsize):
        # list of all particles and class labels that we collected when we fetched the dataset
        num_objs = len(annotated_list)
        list_class_labels = []
        for i in range(0, num_objs):
            list_class_labels.append(annotated_list[i]['label'])

        # a unique list of all possible labels
        unique_labels = np.unique(list_class_labels)

        # imbalanced data affects the performance, we sample equally through all classes
        class_idx = []
        num_sample_class = int(np.floor((bsize / len(unique_labels))))
        for lbl in unique_labels:
            class_idx.append(np.random.choice(np.array(np.nonzero(np.array(list_class_labels) == lbl))[0],
                                              num_sample_class))

        class_idx = np.concatenate(class_idx)
        return class_idx

    def fetch_batch(self, b, e, bsize, flag_new_epoch, flag_new_batch):
        """
        this function fetches the patches from the current tomo based on the batch index
        """
        bstart = b * self.obj.batch_size
        bend = (b * self.obj.batch_size) + self.obj.batch_size

        if b == 0:
            print("********** " + flag_new_batch + " **********")

        mid_dim = int(np.floor(self.obj.patch_size / 2))
        # total_num_samples = len(self.list_annotations)
        num_train_samples = int(np.round(len(self.list_annotations) * (1-self.obj.vald_prc)))
        # num_valid_samples = len(self.list_annotations) - num_train_samples

        if flag_new_epoch:
            # shuffle list of all samples so in the new epoch we get different train and valid samples
            random.shuffle(self.list_annotations)

        Unik_ids = np.unique([sub['tomo_idx'] for sub in self.list_annotations[bstart:bend]])
        self.fetch_tomos(Unik_ids)
        # list_annts = [z for z in (x for x in self.list_annotations if x['tomo_idx'] in Unik_ids)]
        if flag_new_batch == "Train":
            self.train_samples = self.list_annotations[bstart:bend]  # [0:num_train_samples]
        else:
            self.valid_samples = self.list_annotations[bstart:bend]  # [num_train_samples:-1]

        batch_tomo = np.zeros((bsize, self.obj.patch_size, self.obj.patch_size, self.obj.patch_size, 1), dtype=np.float32)

        if self.obj.train_type == "segmentation":
            batch_mask = np.zeros((bsize, self.obj.patch_size, self.obj.patch_size, self.obj.patch_size, self.obj.classNum))
        else:
            batch_mask = np.zeros((bsize, self.obj.patch_size, self.obj.patch_size, self.obj.patch_size, 1), dtype=np.float32)

        # balanced_data_flag = False
        # flag_save = False
        # if balanced_data_flag:
        #     obj_list = self.get_balance_data(self.list_annotations, bsize)
        # else:
        #     obj_list = range(0, len(self.list_annotations))
        cnt = 0
        batch_tomo_cls = []
        # list_of_val_samples = []

        for i in range(0, bsize):
            if flag_new_batch == "Train":
                samples_list = self.train_samples
            else:
                samples_list = self.valid_samples

            tomo_idx = int(samples_list[i]['tomo_idx'])
            sorted_ids = sorted(Unik_ids)
            new_idx = sorted_ids.index(tomo_idx)
            tomo_idx = new_idx
            batch_tomo_cls.append(int(samples_list[i]['label']))

            sample_tomo = self.patches_tomos[tomo_idx]
            sample_mask = self.patches_masks[tomo_idx]


            # Get patch position:
            x, y, z = get_patch_position(self.patches_tomos[tomo_idx].shape, mid_dim, samples_list[i], 0)

            # extract the patch:
            patch_tomo = sample_tomo[z - mid_dim:z + mid_dim, y - mid_dim:y + mid_dim, x - mid_dim:x + mid_dim]
            patch_tomo = (patch_tomo - np.mean(patch_tomo)) / np.std(patch_tomo)

            patch_mask = sample_mask[z - mid_dim:z + mid_dim, y - mid_dim:y + mid_dim, x - mid_dim:x + mid_dim]
            patch_mask = (patch_mask - np.min(patch_mask)) / (np.max(patch_mask) - np.min(patch_mask))
            # vmin, vmax = np.quantile(patch_mask, q=(0.20, 0.80))
            # patch_mask = exposure.rescale_intensity(
            #     patch_mask,
            #     in_range=(vmin, vmax),
            #     out_range=np.float32
            # )
            if e == 0 and flag_new_batch == "Train" and b==0:
                tmp_dir = '/media/noushin/Data/Cryo-ET/DeepET/data2/temp/real_training_patch'
                write_mrc2(np.flip(patch_tomo, axis=1), os.path.join(tmp_dir, f'tomo_patch_{i}_{tomo_idx}_{x}_{y}_{z}.mrc'))
                write_mrc2(np.flip(patch_mask, axis=1), os.path.join(tmp_dir, f'mask_patch_{i}_{tomo_idx}_{x}_{y}_{z}.mrc'))
            # save_npy(patch_mask, self.output_path, "ground", "truth")
            batch_tomo[cnt, :, :, :, 0] = patch_tomo
            # convert masks to categorical labels for classification/segmentation but not for regression
            if self.obj.train_type == "segmentation":
                patch_mask_onehot = to_categorical(patch_mask, self.obj.classNum)
                batch_mask[cnt] = patch_mask_onehot
            else:
                batch_mask[cnt, :, :, :, 0] = patch_mask

            cnt = cnt + 1
        save_csv(batch_tomo_cls, self.output_path, "Train", "Labels")
        if self.obj.augm_prc > 0:
            augmented_percentage = int(bsize * self.obj.augm_prc)
            batch_tomo_augmented = np.zeros((augmented_percentage, self.obj.patch_size,
                                             self.obj.patch_size, self.obj.patch_size, 1))
            batch_mask_augmented = np.zeros((augmented_percentage, self.obj.patch_size,
                                             self.obj.patch_size, self.obj.patch_size, self.obj.classNum))

            batch_tomo = np.vstack((batch_tomo, batch_tomo_augmented))
            batch_mask = np.vstack((batch_mask, batch_mask_augmented))
            augmented_cnt = 0
            for i in range(0, augmented_percentage):
                selected_patch = random.randint(0, bsize - 1)
                selected_tomo = batch_tomo[selected_patch]
                selected_mask = batch_mask[selected_patch]
                # write_mrc2(selected_tomo[:, :, :, 0], '/media/noushin/Data/Cryo-ET/DeepET/data2/results/Metrics/Augmentation/tomo_Original.mrc')

                # rotating by 180 degree horizontally
                batch_tomo[cnt] = np.rot90(selected_tomo, k=2, axes=(0, 2))
                batch_mask[cnt] = np.rot90(selected_mask, k=2, axes=(0, 2))
                # pt_tensor = batch_mask[cnt][:, :, :, 1]
                # rb_tensor = batch_mask[cnt][:, :, :, 2]
                # rb_tensor[rb_tensor == 1] = 2
                # mask_tensor = pt_tensor + rb_tensor
                # write_mrc2(batch_tomo[cnt][:, :, :, 0], '/media/noushin/Data/Cryo-ET/DeepET/data2/results/Metrics/Augmentation/Augmented_tomo_HR.mrc')
                # write_mrc2(mask_tensor, '/media/noushin/Data/Cryo-ET/DeepET/data2/results/Metrics/Augmentation/Augmented_mask_HR.mrc')

                # # rotating by 180 degree vertically
                # batch_tomo[cnt+1] = np.rot90(selected_tomo, k=2, axes=(0, 1))
                # batch_mask[cnt+1] = np.rot90(selected_mask, k=2, axes=(0, 1))
                # pt_tensor = batch_mask[cnt+1][:, :, :, 1]
                # rb_tensor = batch_mask[cnt+1][:, :, :, 2]
                # rb_tensor[rb_tensor == 1] = 2
                # mask_tensor = pt_tensor + rb_tensor
                # write_mrc2(batch_tomo[cnt+1][:, :, :, 0], '/media/noushin/Data/Cryo-ET/DeepET/data2/results/Metrics/Augmentation/Augmented_tomo_VR.mrc')
                # write_mrc2(mask_tensor, '/media/noushin/Data/Cryo-ET/DeepET/data2/results/Metrics/Augmentation/Augmented_mask_VR.mrc')

                # flipping by Z
                # batch_tomo[cnt+2] = np.flip(selected_tomo, axis=0)
                # batch_mask[cnt+2] = np.flip(selected_mask, axis=0)
                # pt_tensor = batch_mask[cnt+2][:, :, :, 1]
                # rb_tensor = batch_mask[cnt+2][:, :, :, 2]
                # rb_tensor[rb_tensor == 1] = 2
                # mask_tensor = pt_tensor + rb_tensor
                # write_mrc2(batch_tomo[cnt+2][:, :, :, 0], '/media/noushin/Data/Cryo-ET/DeepET/data2/results/Metrics/Augmentation/Augmented_tomo_ZFlip.mrc')
                # write_mrc2(mask_tensor, '/media/noushin/Data/Cryo-ET/DeepET/data2/results/Metrics/Augmentation/Augmented_mask_ZFlip.mrc')


                # changing brightness
                """
                Changing the brighness of a image using power-law gamma transformation.
                Gain and gamma are chosen randomly for each image channel.

                Gain chosen between [0.8 - 1.2]
                Gamma chosen between [0.8 - 1.2]

                new_im = gain * im^gamma
                """
                # tomo_new = np.zeros(batch_tomo[selected_patch].shape)
                # for c in range(tomo_new.shape[-1]):
                #     im = selected_tomo[:, :, :, c]
                #     gain, gamma = (1.2 - 0.8) * np.random.random_sample(2, ) + 0.8
                #     im_new = np.sign(im) * gain * (np.abs(im) ** gamma)
                #     tomo_new[:, :, :, c] = im_new
                #     tomo_new = (tomo_new - np.mean(tomo_new)) / np.std(tomo_new)
                #     batch_tomo[cnt+3] = tomo_new
                #     batch_mask[cnt+3] = selected_mask
                #     write_mrc2(batch_tomo[cnt+3][:, :, :, 0], '/media/noushin/Data/Cryo-ET/DeepET/data2/results/Metrics/Augmentation/Augmented_tomo_Brightness.mrc')

                # elastic deformation augmentation
                # [tomo_new, mask_new] = elasticdeform.deform_random_grid([selected_tomo, selected_mask],
                #                                                         sigma=2, axis=[(0, 1, 2), (0, 1, 2)],
                #                                                         order=[1, 0], mode='constant')
                # tomo_new = (tomo_new - np.mean(tomo_new)) / np.std(tomo_new)
                # batch_tomo[cnt] = tomo_new
                # batch_mask[cnt] = mask_new
                # pt_tensor = mask_new[:, :, :, 1]
                # rb_tensor = mask_new[:, :, :, 2]
                # rb_tensor[rb_tensor == 1] = 2
                # mask_tensor = pt_tensor + rb_tensor
                # write_mrc2(batch_tomo[cnt][:, :, :, 0], '/media/noushin/Data/Cryo-ET/DeepET/data2/results/Metrics/Augmentation/Augmented_tomo_Elastic.mrc')
                # write_mrc2(mask_tensor, '/media/noushin/Data/Cryo-ET/DeepET/data2/results/Metrics/Augmentation/Augmented_mask_Elastic.mrc')

                # Noise Injection
                # tomo_new = random_noise(selected_tomo, mode='gaussian', mean=0, var=1, clip=True)
                # batch_tomo[cnt+5] = tomo_new
                # batch_mask[cnt+5] = selected_mask
                # write_mrc2(batch_tomo[cnt + 5][:, :, :, 0], '/media/noushin/Data/Cryo-ET/DeepET/data2/results/Metrics/Augmentation/Augmented_tomo_Noise.mrc')

                # Contrast
                # brightness = 10
                # contrast = random.randint(40, 100)
                # tomo_new = selected_tomo
                # tomo_new = tomo_new * (contrast / 127 + 1) - contrast + brightness
                # tomo_new = (tomo_new - np.mean(tomo_new)) / np.std(tomo_new)
                # batch_tomo[cnt+6] = tomo_new
                # batch_mask[cnt+6] = selected_mask
                # write_mrc2(batch_tomo[cnt+6][:, :, :, 0], '/media/noushin/Data/Cryo-ET/DeepET/data2/results/Metrics/Augmentation/Augmented_tomo_Contrast.mrc')

                # Blurring using TorchIO package
                # blur = tio.RandomBlur()
                # tomo_new = blur(selected_tomo)
                # batch_tomo[cnt] = tomo_new
                # batch_mask[cnt] = selected_mask
                # write_mrc2(batch_tomo[cnt + 7][:, :, :, 0],'/media/noushin/Data/Cryo-ET/DeepET/data2/results/Metrics/Augmentation/Augmented_tomo_blurred.mrc')

                augmented_cnt = augmented_cnt + 1
                cnt = cnt + 1
        return batch_tomo, batch_mask

    def generate_background(self, thistomo, thismask):
        label_indices = np.argwhere(thismask != 0)
        bg_indx = np.argwhere(thismask == 0)
        cnt = 0

        for indx in label_indices:
            thistomo[tuple(indx)] = thistomo[tuple(bg_indx[cnt])]
            cnt = cnt + 1
        return thistomo

    def realtime_output(self, newstr):
        self.printstr = self.printstr + newstr
        self.obj.ui.textEdit.setText(self.printstr)

    def print_history(self, history):
        printstr = ""
        indxcol = 1
        for key, value in history.history.items():
            printstr = printstr + str(key) + " : " + str(value) + " | "
            if indxcol % 50 == 0:
                printstr = printstr + "\n"
            indxcol = indxcol + 1

        return printstr

    def save_history(self, batch_tomo):
        # serialize model to JSON
        model_json = self.net.to_json()
        with open(os.path.join(self.output_path, "model.json"), "w") as json_file:
            json_file.write(model_json)

        if self.obj.train_type == "segmentation":
            self.save_layer_output(batch_tomo, layer_name="cls_layer")
        else:
            self.save_layer_output(batch_tomo, layer_name="reg_layer")

        save_csv(self.history_train_acc, self.output_path, "Train", "Accuracy_Details")
        save_csv(self.history_vald_acc, self.output_path, "Validation", "Accuracy_Details")
        save_csv(self.history_train_loss, self.output_path, "Train", "Loss_Details")
        save_csv(self.history_vald_loss, self.output_path, "Validation", "Loss_Details")
        save_csv(self.history_lr, self.output_path, "Train", "LearningRate_Details")
        if self.obj.train_type == "segmentation":
            save_csv(self.history_f1_score, self.output_path, "Validation", "F1Score_Details")
            save_csv(self.history_precision, self.output_path, "Validation", "Precision_Details")
        save_csv(self.history_recall, self.output_path, "Validation", "Recall_Details")

        # averaging the accuracy and loss over all folds
        self.train_acc = np.mean(self.history_train_acc, axis=1)
        self.vald_acc = np.mean(self.history_vald_acc, axis=1)
        self.train_loss = np.mean(self.history_train_loss, axis=1)
        self.vald_loss = np.mean(self.history_vald_loss, axis=1)
        if self.obj.train_type == "segmentation":
            self.f1_score = np.mean(self.history_f1_score, axis=1)
            self.precision = np.mean(self.history_precision, axis=1)
            self.recall = np.mean(self.history_recall, axis=1)

        # saving the average results from folds
        save_csv(self.train_acc, self.output_path, flag="Train", name="Averaged_Accuracy")
        save_csv(self.vald_acc, self.output_path, flag="Validation", name="Averaged_Accuracy")
        save_csv(self.train_loss, self.output_path, flag="Train", name="Averaged_Loss")
        save_csv(self.vald_loss, self.output_path, flag="Validation", name="Averaged_Loss")
        if self.obj.train_type == "segmentation":
            save_csv(self.f1_score, self.output_path, flag="Validation", name="Averaged_F1")
            save_csv(self.precision, self.output_path, flag="Validation", name="Averaged_Precision")
            save_csv(self.recall, self.output_path, flag="Validation", name="Averaged_Recall")

    def plots(self):
        start_point = 0  # dropping the first few point in plots due to unstable behavior of model

        plt.figure(num=1, figsize=(8, 6), dpi=100)
        plot_train_vs_vald(self.train_loss[start_point:], self.vald_loss[start_point:],
                           self.output_path, self.obj.epochs, is_loss=True)

        if self.obj.train_type == "segmentation":
            plt.figure(num=2, figsize=(8, 6), dpi=100)
            plot_train_vs_vald(self.train_acc[start_point:], self.vald_acc[start_point:],
                               self.output_path, self.obj.epochs)

        # plt.figure(num=3, figsize=(8, 6), dpi=100)
        # plot_lr(self.lr_history['iterations'], self.lr_history['lr'], self.output_path, self.obj.epochs)

        # plt.plot(self.lr_history['iterations'], self.lr_history['lr'])
        if self.obj.train_type == "segmentation":
            general_plot(self.f1_score, self.output_path, ('F1 Score', 'epochs'),
                         self.obj.class_names, self.obj.epochs, 4)
            general_plot(self.precision, self.output_path, ('Precision', 'epochs'),
                         self.obj.class_names, self.obj.epochs, 5)
            general_plot(self.recall, self.output_path, ('Recall', 'epochs'),
                        self.obj.class_names, self.obj.epochs, 6)
        # else:
        #     general_plot(self.recall, self.output_path, ('R2-Score', 'epochs'),
        #                  self.obj.class_names, self.obj.epochs, 7)

    def save(self):
        hyperparameter_setting = self.collect_results()
        with open(os.path.join(self.output_path, "HyperParameters.txt"), "w") as text_file:
            text_file.write(hyperparameter_setting)
        print(hyperparameter_setting)

        shutil.copyfile(os.path.join(ROOT_DIR, "code/train_models.py"),
                        os.path.join(self.output_path, "train_models.txt"))

        shutil.copyfile(os.path.join(ROOT_DIR, "utils/models.py"),
                        os.path.join(self.output_path, "models.txt"))

    def set_optimizer(self):
        self.optimizer = Adam(learning_rate=self.lr, beta_1=.9, beta_2=.999, epsilon=1e-08, decay=0.0, clipnorm=10)


        if self.obj.opt == "SGD":
            self.optimizer = SGD(learning_rate=self.lr, decay=0.0, momentum=0.9, nesterov=True)
        elif self.obj.opt == "Adagrad":
            self.obj.optimizer = Adagrad(learning_rate=self.lr, epsilon=1e-08, decay=0.0)
        elif self.obj.opt == "Adadelta":
            self.optimizer = Adadelta(learning_rate=self.lr, rho=0.95, epsilon=1e-08, decay=0.0)
        elif self.obj.opt == "Adamax":
            self.optimizer = Adamax(learning_rate=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        elif self.obj.opt == "Nadam":
            self.optimizer = Nadam(learning_rate=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        elif self.obj.opt == "RMSprop":
            self.optimizer = RMSprop(learning_rate=self.lr, rho=0.9, epsilon=1e-08, decay=0.0)

    # learning rate schedule
    def lr_decay_step(self, epoch):
        lr = self.initial_lr * math.pow(.9, math.floor((1 + epoch) / 5))
        return float(lr)

    def lr_decay_exp(self, epoch):
        # compute the learning rate for the current epoch
        exp = np.floor((1 + epoch) / 10)
        lr = self.initial_lr * (.25 ** exp)
        return float(lr)

    def lr_decay_poly(self, epoch):
        decay = (1 - (epoch / float(100))) ** 1.0
        lr = self.initial_lr * decay
        return float(lr)

    def lr_cyclic(self, base_lr=0.0001, max_lr=0.0003, step_size=48., mode='triangular',
                  gamma=1., scale_fn=None, scale_mode='cycle'):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        if scale_fn is None:
            if mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self._cycliclr_reset()

    def _cycliclr_reset(self, new_base_lr=None, new_max_lr=None, new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * \
                   np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)

    def on_train_begin(self):
        if self.clr_iterations == 0:
            bk.set_value(self.net.optimizer.lr, self.base_lr)
        else:
            bk.set_value(self.net.optimizer.lr, self.clr())

    def on_batch_end(self, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.lr_history.setdefault('lr', []).append(bk.get_value(self.net.optimizer.lr))
        self.lr_history.setdefault('iterations', []).append(self.trn_iterations)

        for kkey, vval in logs.items():
            self.lr_history.setdefault(kkey, []).append(vval)

        bk.set_value(self.net.optimizer.lr, self.clr())

    def set_lr(self, epoch):
        """schedule learning rate decay using different methods
        step Decay: drops learning rate by a factor as a step function
        exp decay: drops learning rate exponentially
        poly decay: drops learning rate by a polynomial function
        cyclic: drops learning rate by a cyclic method"""
        if self.lr_type == "Fixed":
            k.set_value(self.net.optimizer.lr, self.lr)
        elif self.lr_type == "Step":
            k.set_value(self.net.optimizer.lr, self.lr_decay_step(epoch))
            # return self.lr_decay_step(epoch)
        elif self.lr_type == "Exponential":
            k.set_value(self.net.optimizer.lr, self.lr_decay_exp(epoch))
        elif self.lr_type == "Polynomial":
            k.set_value(self.net.optimizer.lr, self.lr_decay_poly(epoch))
        elif self.lr_type == "Cyclic":
            return self.lr_cyclic(base_lr=self.lr, max_lr=0.0003, step_size=222.,
                                  mode='triangular', gamma=1., scale_fn=None, scale_mode='cycle')
        else:
            ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=1, min_lr=1e-07, mode='min', verbose=1)

    def set_compile(self):
        # self.obj.metrics = ['accuracy', self.lr]
        # keras.metrics.TruePositives(name='tp'),
        # keras.metrics.FalsePositives(name='fp'),
        # keras.metrics.TrueNegatives(name='tn'),
        # keras.metrics.FalseNegatives(name='fn'),
        self.obj.metrics = [metrics.BinaryAccuracy(name='acc'),
                            metrics.Precision(name='precision'),
                            metrics.Recall(name='recall'),
                            metrics.AUC(name='auc')]
        s = Semantic_loss_functions()
        if self.obj.loss == "tversky":
            self.net.compile(optimizer=self.optimizer, loss=s.tversky_loss, metrics=['accuracy'])
        elif self.obj.loss == "focal_loss":
            self.net.compile(optimizer=self.optimizer, loss=s.focal_loss, metrics=['accuracy'])
        elif self.obj.loss == "dice_loss":
            self.net.compile(optimizer=self.optimizer, loss=s.dice_loss, metrics=['accuracy'])
        elif self.obj.loss == "bce_dice_loss":
            self.net.compile(optimizer=self.optimizer, loss=s.bce_dice_loss, metrics=['accuracy'])
        elif self.obj.loss == "focal_tversky":
            self.net.compile(optimizer=self.optimizer, loss=s.focal_tversky, metrics=['accuracy'])
        elif self.obj.loss == "mse":
            self.net.compile(optimizer=self.optimizer, loss=losses.MeanSquaredError(), metrics=['mae'])
        elif self.obj.loss == "mae":
            self.net.compile(optimizer=self.optimizer, loss=losses.MeanAbsoluteError(), metrics=['mae'])
        elif self.obj.loss == "huber":
            self.net.compile(optimizer=self.optimizer, loss=losses.Huber(delta=1.5), metrics=['mae'])
        elif self.obj.loss == "quantile":
            perc_points = [0.01, 0.25, 0.5, 0.75, 0.99]
            self.net.compile(optimizer=self.optimizer, loss=s.QuantileLoss(perc_points, delta=1.5), metrics=['mae'])
        else:
            self.net.compile(optimizer=self.optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

    def set_weight_callback(self, e):
        # checkpoint directory
        checkpoint_dir = os.path.join(self.output_path, 'weights-improvement-' + str(e) + '.h5')
        # callbacklist=[TensorBoard(log_dir=log_folder)]

        self.net.save(checkpoint_dir)

    def collect_results(self):
        # TODO: add calculation of union of interest in plots file.
        list_imgs = glob(os.path.join(self.train_img_path, "*.mrc"))
        list_imgs.sort(key=lambda ff: int(re.sub('\D', '', ff)))

        setting_info = "Saving folder Path =" + str(self.output_path)
        setting_info = setting_info + "\nData Path = " + str(self.train_img_path)
        setting_info = setting_info + "\nList of images:\n" + str(list_imgs)
        setting_info = setting_info + "\nNumber of Epochs In Training = " + str(self.obj.epochs)
        setting_info = setting_info + "\nBatch Size = " + str(self.obj.batch_size)
        setting_info = setting_info + "\nPatch Size = " + str(self.obj.patch_size)
        setting_info = setting_info + "\nInitial Learning Rate = " + str(self.obj.lr)
        setting_info = setting_info + "\nDecayed Learning Rate = " + str(k.eval(self.net.optimizer.lr))
        setting_info = setting_info + "\nTrain accuracy = " + str(np.mean(self.train_acc))
        setting_info = setting_info + "\nTrain loss = " + str(np.mean(self.train_loss))
        setting_info = setting_info + "\nValidation accuracy = " + str(np.mean(self.vald_acc))
        setting_info = setting_info + "\nValidation loss = " + str(np.mean(self.vald_loss))
        setting_info = setting_info + "\nAverage F1 Score = " + str(np.mean(self.f1_score))
        setting_info = setting_info + "\nLoss Function = " + str(self.obj.loss)
        setting_info = setting_info + "\nOptimizer = " + str(self.obj.opt)
        setting_info = setting_info + "\nLearning rate Function = " + str(self.lr_type)
        setting_info = setting_info + "\nAugmentation Percentage = " + str(self.obj.augm_prc)
        # setting_info = setting_info + "\nAugmentation Type = " + str(self.obj.augm_type)
        setting_info = setting_info + "\nValdiation Percentage = " + str(self.obj.vald_prc)
        setting_info = setting_info + "\nProcess Time in seconds = " + str(self.process_time)
        return setting_info

    def save_layer_output(self, xinput, layer_name="cls_layer"):
        intermediate_layer_model = Model(inputs=self.net.input, outputs=self.net.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(xinput)
        filename = str(layer_name) + "_Features"
        np.save(os.path.join(self.output_path, filename), intermediate_output)

    def save_layer_filter(self, ith_layer, layer_name="cls_layer"):
        filters, biases = ith_layer.get_weights()
        # print(ith_layer.name, filters.shape)
        filename = str(layer_name) + "_Filters"
        np.save(os.path.join(self.output_path, filename), filters)
