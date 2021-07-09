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

import h5py
import time
import math
import shutil
import random
import string
from utils.plots import *
from utils.utility_tools import *
from utils.CyclicLR.clr_callback import CyclicLR
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import *
from tensorflow.keras import Model, callbacks  # backend, ,
from tensorflow.keras import layers
from tensorflow.keras import backend as bk
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import preprocessing


# loading batches of data
class DataPreparation(keras.utils.Sequence):
    """we iterate over the data as numpy arrays"""

    def __init__(self, obj):
        self.tomos = None
        self.masks = None
        self.dataset = None
        self.batched_dataset = None


        self.batch_size = obj.batch_size
        self.patch_size = obj.patch_size
        self.img_size = obj.img_dim
        self.train_img_path = os.path.join(obj.base_path, "images/train/")
        self.train_target_path = os.path.join(obj.base_path, "targets/train/")

        seednum = 2
        np.random.seed(seednum)
        random_str = ''.join([random.choice(string.ascii_uppercase + string.digits) for n in range(16)])
        self.output_path = os.path.join(obj.output_path, random_str)
        os.makedirs(self.output_path)

        # check values
        is_positive(self.batch_size, 'batch_size')
        is_positive(self.patch_size, 'patch_size')
        is_dir(self.train_img_path)
        is_dir(self.train_target_path)
        is_dir(self.output_path)

        # load the train and validation data
        # self.get_tomo_list()
        self.generate_batches()

    # def get_tomo_list(self):
    #     """
    #     the idea is to get only the name of all tomogram files and their corresponding targets
    #     """
    #     from glob import glob
    #     self.tomo_name_list = glob(os.path.join(self.train_img_path, "*.mrc"))
    #     self.target_name_list = glob(os.path.join(self.train_target_path, "*.mrc"))
    #     # self.load_data()

    # def load_data(self):
    #     """
    #     loads the whole dataset at once, can make the process slow,
    #     reads the tomogram and target data from the list of names one by one
    #     and adds them to two separate lists called tomo_list and target_list.
    #     """
    #     for idx in range(0, len(self.train_img_path)):
    #         data = read_mrc(self.tomo_name_list[idx])
    #         target = read_mrc(self.target_name_list[idx])
    #
    #         # check if the tomogram and the target has similar shape
    #         is_same_shape(data.shape, target.shape)
    #
    #         self.tomo_list.append(data)
    #         self.target_list.append(target)

    def modify_image(self, type="tomo"):
        """you can add a preprocessing functions here"""

        image = tf.expand_dims(self.tomos, 0)
        image = tf.extract_image_patches(image,
                                        ksizes=[1, self.patch_size, self.patch_size, 1],
                                        strides=[1, self.patch_size, self.patch_size, 1],
                                        rates=[1, 1, 1, 1],
                                        padding='SAME',
                                        name=None)
        image = tf.reshape(image, shape=[-1, self.patch_size, self.patch_size, 1])
        if type != "tomo":
            self.tomos = image
        else:
            self.masks = image

    def parse_function(self):
        self.tomos = tf.read_file(self.tomos)
        self.tomos = tf.image.decode_image(self.tomos)
        self.masks = tf.read_file(self.masks)
        self.masks = tf.image.decode_image(self.masks)
        self.modify_image("tomo")
        self.modify_image("mask")

    def load_dataset(self):
        from glob import glob
        self.tomos = glob(self.train_img_path, "*.mrc")
        self.tomos = tf.constant(self.tomos)
        self.masks = glob(self.train_target_path, "*.mrc")
        self.masks = tf.constant(self.masks)

        self.dataset = tf.data.Dataset.from_tensor_slices((self.tomos, self.masks))
        self.dataset = self.dataset.map(self.parse_function)

    def generate_batches(self):
        """
        fetches the nth set of batches required for patch generation
        """
        self.load_dataset()
        self.batched_dataset = self.dataset.batch(self.batch_size)


class CNNModels:

    def __init__(self, obj):
        self.data = DataPreparation(obj)

        # define values
        self.lrr = []
        self.lrT = None
        self.net = None
        self.history = None
        self.optimizer = None
        self.checkpoint = None
        self.layer_name = None
        self.process_time = None
        self.callbacks = None

        # initialize values
        self.obj = obj
        self.width = self.obj.img_dim[0]
        self.height = self.obj.img_dim[1]
        if self.obj.dim_num > 2:
            self.depth = self.obj.img_dim[2]

        # check values
        is_positive(self.obj.epochs, 'epochs')
        is_positive(self.obj.classNum, 'num_class')

        # initialize model
        self.train_model()

    def train_model(self):
        """
        This function starts the training procedure by calling
        different built-in functions of the class CNNModel
        """
        self.get_model()
        self.fit_model()
        self.plots()
        self.save()
        plt.show(block=True)

    def set_optimizer(self):
        self.optimizer = Adam(lr=self.obj.lr, beta_1=.9, beta_2=.999, epsilon=1e-08, decay=0.0)

        if self.obj.opt == "SGD":
            self.optimizer = SGD(lr=self.obj.lr, decay=0.0, momentum=0.9, nesterov=True)
        elif self.obj.opt == "Adagrad":
            self.obj.optimizer = Adagrad(lr=self.obj.lr, epsilon=1e-08, decay=0.0)
        elif self.obj.opt == "Adadelta":
            self.optimizer = Adadelta(lr=self.obj.lr, rho=0.95, epsilon=1e-08, decay=0.0)
        elif self.obj.opt == "Adamax":
            self.optimizer = Adamax(lr=self.obj.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        elif self.obj.opt == "Nadam":
            self.optimizer = Nadam(lr=self.obj.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        elif self.obj.opt == "RMSprop":
            self.optimizer = RMSprop(lr=self.obj.lr, rho=0.9, epsilon=1e-08, decay=0.0)

    # learning rate schedule
    def step_decay(self):
        initial_lrate = self.obj.lr
        drop = 0.5
        epochs_drop = 50
        lrate = initial_lrate * math.pow(drop, math.floor((1 + self.obj.epochs) / epochs_drop))
        self.obj.lrr.append(lrate)
        return lrate

    def set_lr(self):
        # learning schedule callback
        self.obj.lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=10,
                                        min_lr=1e-06, mode='min', verbose=1)
        if self.lrT == "step_decay":
            self.obj.lr = LearningRateScheduler(self.step_decay())
        elif self.lrT == "lambda":
            self.obj.lr = LearningRateScheduler(lambda this_epoch: self.lr * 0.99 ** this_epoch)
        elif self.lrT == "cyclic":
            self.obj.lr = CyclicLR(base_lr=self.obj.lr, max_lr=6e-04, step_size=500., mode='exp_range', gamma=0.99994)

    def get_lr_metric(self):
        def lr(y_true, y_pred):
            # lrr.append(float(K.get_value(optimizer.lr)))
            self.obj.lr = self.optimizer.lr
        return self.obj.lr

    def set_compile(self):
        # self.get_lr_metric()
        # self.obj.metrics = ['accuracy', self.obj.lr]
        self.obj.metrics = ['accuracy']
        # self.obj.metrics = [keras.metrics.TruePositives(name='tp'),
        #                     keras.metrics.FalsePositives(name='fp'),
        #                     keras.metrics.TrueNegatives(name='tn'),
        #                     keras.metrics.FalseNegatives(name='fn'),
        #                     keras.metrics.BinaryAccuracy(name='accuracy'),
        #                     keras.metrics.Precision(name='precision'),
        #                     keras.metrics.Recall(name='recall'),
        #                     keras.metrics.AUC(name='auc'),
        #                     self.get_lr_metric()
        #                     ]
        if self.obj.loss != "tversky":
            self.net.compile(optimizer=self.optimizer, loss=self.obj.loss, metrics=[self.obj.metrics])
        else:
            self.net.compile(optimizer=self.optimizer, loss=self.tversky_loss, metrics=[self.obj.metrics])

    def set_checkpoint(self):
        self.checkpoint = ModelCheckpoint(self.data.output_path, monitor='val_acc',
                                          verbose=1, save_best_only=True, mode='max')

    def set_callback(self):
        # callbacks_list = [self.checkpoint, self.obj.lr]
        # self.callbacks = "[checkpoint, reduce_lr]"
        self.callbacks = [callbacks.ModelCheckpoint("weights.h5", save_best_only=True)]

    def get_model(self):
        if self.obj.model_type == "2D UNet":
            self.net = self.unet2d()
        elif self.obj.model_type == "3D UNet":
            self.net = self.unet3d()

        # set the properties of the mdoel
        self.set_optimizer()
        self.set_compile()
        print(self.net.summary())

    def fit_model(self):
        start = time.clock()
        batch_idx = 0
        for e in range(self.obj.epochs):
            # to collect training outputs
            list_loss_train = []
            list_acc_train = []

            # dataset =  image_dataset_from_directory(self.obj.base_path,
            #                                 label_mode = None,
            #                                 seed = 1,
            #                                 subset = 'training',
            #                                 validation_split = 0.1,
            #                                 image_size = (900, 900))
            # batch_tomo: numpy array [batch_idx, z, y, x, channel] in our case only 1 channel
            # batch_target: numpy array [batch_idx, z, y, x, class_idx] is one-hot encoded
            batch_data = np.zeros((self.obj.batch_size, self.width, self.height, self.depth, 1))
            batch_target = np.zeros((self.obj.batch_size, self.width, self.height, self.depth, self.obj.classNum))

            # self.history = self.net.fit(self.train_data[train], self.train_labels_one_hot_coded,
            #                     epochs=self.epoch, batch_size=self.data.batch_size, shuffle=True,
            #                     validation_data=(self.train_data[vald], self.vald_labels_one_hot_coded),
            #                     callbacks=self.callbacks_list)
        end = time.clock()
        self.process_time = (end - start)

    def plots(self):
        start_point = 10  # dropping the first few point in plots due to unstable behavior of model
        # cnf_matrix = np.zeros(shape=[self.obj.classNum, self.obj.classNum])
        # figure(num=1, figsize=(8, 6), dpi=80)
        # plot_folds_accuracy(model_history)

        # plt.figure(num=2, figsize=(8, 6), dpi=80)
        # plot_folds_loss(model_history)

        plt.figure(num=1, figsize=(8, 6), dpi=100)
        plot_train_vs_vald(self.train_loss[start_point:], self.vald_loss[start_point:],
                           self.obj.output_path, self.obj.epochs, is_loss=True)

        plt.figure(num=2, figsize=(8, 6), dpi=100)
        plot_train_vs_vald(self.train_acc[start_point:], self.vald_acc[start_point:],
                           self.obj.output_path, self.obj.epochs)

        plt.figure(num=3, figsize=(8, 6), dpi=100)
        plot_lr(self.train_lr[start_point:], self.obj.output_path, self.obj.epochs)

        # Plot all ROC curves
        plt.figure(num=4, figsize=(8, 6), dpi=100)
        plot_roc(self.test_labels_one_hot_coded, self.test_predicted_probs,
                 self.obj.classNum, self.obj.output_path, self.obj.epochs)

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(self.test_labels, self.test_predicted_labels)
        np.set_printoptions(precision=2)

        cnf_matrix2 = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print(np.average(cnf_matrix2.diagonal()))

        # Plot and save non-normalized confusion matrix
        plt.figure(num=5, figsize=(5, 5), dpi=100)
        plot_confusion_matrix(cnf_matrix, self.obj.classNum, self.obj.output_path, self.obj.epochs)

        # Plot normalized confusion matrix
        plt.figure(num=6, figsize=(5, 5), dpi=100)
        plot_confusion_matrix(cnf_matrix, self.obj.classNum, self.obj.output_path, self.obj.epochs, normalize=True)

        cnf_matrix2 = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print(np.average(cnf_matrix2.diagonal()))

    def save(self):

        # serialize model to JSON
        model_json = self.net.to_json()
        with open(os.path.join(self.data.output_path, "model.json"), "w") as json_file:
            json_file.write(model_json)

        # evaluation on train
        # train_loss, train_acc, train_lr = self.net.evaluate(train_data, train_labels_one_hot_coded, batch_size=1)
        # print(train_loss, train_acc, train_lr)
        #
        # train_predicted_probs = self.net.predict(train_data, batch_size=1)
        # train_predicted_labels = train_predicted_probs.argmax(axis=-1)
        #
        # # Saving Train results
        # self.save_npy(train_predicted_probs, flag="Train", name="Probabilities")
        # self.save_npy(train_predicted_labels, flag="Train", name="ClassLabels")
        # self.save_csv(train_predicted_probs, flag="Train", name="Probabilities")
        # self.save_csv(train_predicted_labels, flag="Train", name="ClassLabels")
        #
        # # evaluation on Test
        # test_loss, test_acc, test_lr = self.net.evaluate(test_data, test_labels_one_hot_coded, batch_size=1)
        # print(test_loss, test_acc, test_lr)

        hyperparameter_setting = ""  # self.collect_results()
        with open(os.path.join(self.data.output_path, "HyperParameters.txt"), "w") as text_file:
            text_file.write(hyperparameter_setting)
        print(hyperparameter_setting)

        shutil.copyfile(os.path.join(self.data.output_path, "models.py"),
                        os.path.join(self.data.output_path, "models.txt"))

    # saving labels or predicted probablities as a npy file
    def save_npy(self, data, flag="Train", name="Probabilities"):
        np.save(os.path.join(self.data.output_path,
                             flag + "_" + name + "_" + str(self.obj.epochs) + "_Epochs.npy"), data)

    # saving labels or predicted probablities as a csv file
    def save_csv(self, data, flag="Train", name="Probabilities"):
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.data.output_path, flag + "_" + name + "_" + str(self.obj.epochs) + "_Epochs.csv"))

    def save_layer_output(self, x, name="Train"):
        intermediate_layer_model = Model(inputs=self.net.input, outputs=self.net.get_layer(self.layer_name).output)
        intermediate_output = intermediate_layer_model.predict(x)
        filename = name + "_fc6_Layer_Features"
        np.save(os.path.join(self.data.output_path, filename), intermediate_output)

    def collect_results(self):
        # TODO: add calculation of union of interest in plots file.
        setting_info = "Saving folder Path =" + str(self.data.output_path)
        setting_info = setting_info + "\nData Path = " + str(self.data.train_img_path)
        setting_info = setting_info + "\nNumber of Epochs In Training = " + str(self.obj.epochs)
        setting_info = setting_info + "\nBatchsize = " + str(self.data.batch_size)
        setting_info = setting_info + "\nLearning Rate = " + str(self.obj.lr)
        setting_info = setting_info + "\nFeatures Saved For Layer = " + str(self.layer_name)
        setting_info = setting_info + "\nCallbacks = " + self.callbacks
        setting_info = setting_info + "\nTrain accuracy = " + str(self.train_acc)
        setting_info = setting_info + "\nTrain loss = " + str(self.train_loss)
        setting_info = setting_info + "\nValidation accuracy = " + str(np.mean(self.vald_acc))
        setting_info = setting_info + "\nValidation loss = " + str(np.mean(self.vald_loss))
        setting_info = setting_info + "\nTest accuracy = " + str(self.test_acc)
        setting_info = setting_info + "\nTest loss = " + str(self.test_loss)
        setting_info = setting_info + "\nProcess Time in seconds = " + str(self.process_time)
        return setting_info

    def tversky_loss(y_true, y_pred):
        alpha = 0.5
        beta = 0.5

        ones = bk.ones(bk.shape(y_true))
        p0 = y_pred  # proba that voxels are class i
        p1 = ones - y_pred  # proba that voxels are not class i
        g0 = y_true
        g1 = ones - y_true

        num = bk.sum(p0 * g0, (0, 1, 2, 3))
        den = num + alpha * bk.sum(p0 * g1, (0, 1, 2, 3)) + beta * bk.sum(p1 * g0, (0, 1, 2, 3))

        t_sum = bk.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

        num_classes = bk.cast(bk.shape(y_true)[-1], 'float32')
        return num_classes - t_sum

    def unet2d(self):
        # The original 2D UNET mdoel
        input_img = layers.Input(shape=(self.width, self.height, 1))

        # down-sampling part of the network
        x = layers.Conv2D(32, 3, strides=2, padding="same")(input_img)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same")(previous_block_activation)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # up-sampling part of the network
        for filters in [256, 128, 64, 32]:
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same")(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(self.obj.classNum, 3, activation="softmax", padding="same")(x)

        # Define the model
        model = Model(input_img, outputs)
        return model

    def unet3d(self):
        # The UNET model from DeepFinder
        input_img = layers.Input(shape=(self.width, self.height, self.depth, 1))

        x = layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu')(input_img)
        high = layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)

        x = layers.MaxPooling3D((2, 2, 2), strides=None)(high)

        x = layers.Conv3D(48, (3, 3, 3), padding='same', activation='relu')(x)
        mid = layers.Conv3D(48, (3, 3, 3), padding='same', activation='relu')(x)

        x = layers.MaxPooling3D((2, 2, 2), strides=None)(mid)

        x = layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
        x = layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
        x = layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
        x = layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)

        x = layers.UpSampling3D(size=(2, 2, 2), data_format='channels_last')(x)
        x = layers.Conv3D(64, (2, 2, 2), padding='same', activation='relu')(x)

        x = layers.concatenate([x, mid])
        x = layers.Conv3D(48, (3, 3, 3), padding='same', activation='relu')(x)
        x = layers.Conv3D(48, (3, 3, 3), padding='same', activation='relu')(x)

        x = layers.UpSampling3D(size=(2, 2, 2), data_format='channels_last')(x)
        x = layers.Conv3D(48, (2, 2, 2), padding='same', activation='relu')(x)

        x = layers.concatenate([x, high])
        x = layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
        x = layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)

        output = layers.Conv3D(self.obj.classNum, (1, 1, 1), padding='same', activation='softmax')(x)

        model = Model(input_img, output)
        return model
