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
import re
import time
import math
import shutil
import random
import string
from utils.plots import *
from utils.utility_tools import *
from sklearn.metrics import confusion_matrix
from utils.CyclicLR.clr_callback import CyclicLR
from sklearn.model_selection import StratifiedKFold

from tensorflow import keras
from tensorflow.keras.optimizers import *
from tensorflow.keras import Model  # backend, ,
from tensorflow.keras import layers
from tensorflow.keras import backend as bk
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
# from sklearn.feature_extraction import image
import tensorflow as tf
from sklearn.model_selection import train_test_split

# loading batches of data
class DataPreparation(keras.utils.Sequence):
    """we iterate over the data as numpy arrays"""

    def __init__(self, obj):
        self.list_tomos_IDs = None
        self.list_masks_IDs = None
        self.batch_size = obj.batch_size
        self.patch_size = obj.patch_size
        self.img_size = obj.img_dim
        self.train_img_path = os.path.join(obj.base_path, "images/")
        self.train_target_path = os.path.join(obj.base_path, "targets/")

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
        self.fetch_dataset()

    def fetch_dataset(self):
        from glob import glob
        self.list_tomos_IDs = glob(os.path.join(self.train_img_path, "*.mrc"))
        self.list_masks_IDs = glob(os.path.join(self.train_target_path, "*.mrc"))

        # check if every tomo has a corresponding mask
        if len(self.list_tomos_IDs) != len(self.list_masks_IDs):
            display("Expected two" + str(len(self.list_tomos_IDs)) + ", received " +
                   str(len(self.list_tomos_IDs)) + " and "+ str(len(self.list_masks_IDs)) +
                    ". \n There is a missing pair of tomogram and target.")
            sys.exit()

        self.list_tomos_IDs.sort(key=lambda f: int(re.sub('\D', '', f)))
        self.list_masks_IDs.sort(key=lambda r: int(re.sub('\D', '', r)))


class CNNModels:

    def __init__(self, obj):
        self.data = DataPreparation(obj)

        # define values
        self.net = None
        self.optimizer = None
        self.checkpoint = None
        self.layer_name = None
        self.process_time = None
        self.callbacks = None
        self.model_weight = None

        self.patch_overlap = False
        self.batch_idx = 0
        self.tomo_index = 0
        self.printstr = ""

        # values to collect outputs
        self.model_history = []
        self.batch_mask_onehot = []
        self.history_lr = []
        # self.history_recall = []
        self.history_vald_acc = []
        # self.history_f1_score = []
        self.history_train_acc = []
        self.history_vald_loss = []
        # self.history_precision = []
        self.history_train_loss = []

        # initialize values
        self.obj = obj
        self.lr = self.obj.lr
        self.width = self.obj.img_dim[0]
        self.height = self.obj.img_dim[1]
        if self.obj.dim_num > 2:
            self.depth = self.obj.img_dim[2]

        self.nump_xaxis = int(np.floor(self.width / self.obj.patch_size))
        self.nump_yaxis = int(np.floor(self.height / self.obj.patch_size))
        self.nump_zaxis = int(np.floor(self.depth / 32))

        self.nump_total = self.nump_xaxis * self.nump_yaxis * self.nump_zaxis

        self.batch_tomo = []
        self.batch_mask = []

        # self.batch_tomo = np.zeros((self.nump_total, 25,
        #                             self.obj.patch_size, self.obj.patch_size, 1))
        # self.batch_mask = np.zeros((self.nump_total, 25,
        #                             self.obj.patch_size, self.obj.patch_size, self.obj.classNum))

        # check values
        is_positive(self.obj.epochs, 'epochs')
        is_positive(self.obj.classNum, 'num_class')

        # initialize model
        self.train_model()

    def train_model(self):
        """This function starts the training procedure by calling
           different built-in functions of the class CNNModel
        """
        self.get_model()
        self.fit_model()
        self.plots()
        self.save()
        plt.show(block=True)

    def get_model(self):
        if self.obj.model_type == "2D UNet":
            self.unet2d()
        elif self.obj.model_type == "3D UNet":
            self.unet3d()

        # set the properties of the mdoel
        self.set_optimizer()
        self.set_compile()
        print(self.net.summary())

    def fit_model(self):
        start = time.clock()
        batch_itr = int(np.floor(self.nump_total) / self.obj.batch_size)
        for e in range(self.obj.epochs):
            self.realtime_output("########## Start Epoch {epochnum} ########## \n\n".format(epochnum=e))
            # fetch all patches of the current batch
            for t in range(len(self.data.list_tomos_IDs)):
                self.realtime_output("********** Start Tomo {tnum} ********** \n\n".format(tnum=t))
                self.fetch_tomo()
                for b in range(batch_itr):
                    self.realtime_output("---------- Start Batch {bnum} ---------- \n\n".format(bnum=b))
                    if b == batch_itr:
                        self.batch_idx = 0
                    else:
                        self.batch_idx = b

                    # fetch the current batch of patches
                    self.fetch_batch()

                    # Split the data
                    x_train, x_vald, y_train, y_vald = train_test_split(self.batch_tomo, self.batch_mask_onehot,
                                                                        test_size=0.2, shuffle=True)
                    x_train = np.expand_dims(x_train, axis=4)
                    x_vald = np.expand_dims(x_vald, axis=4)
                    y_train = np.array(y_train)
                    y_vald = np.array(y_vald)
                    # train_loss = self.net.train_on_batch(self.batch_tomo, self.batch_mask_onehot,
                    #                                      class_weight=self.model_weight)
                    self.set_callback()
                    history = self.net.fit(x_train, y_train, epochs=1,
                                           batch_size=1, shuffle=False,
                                           validation_data=(x_vald, y_vald),
                                           callbacks=self.callbacks)

                    self.model_history.append(history)
                    self.history_train_acc.append(history.history['acc'])
                    self.history_vald_acc.append(history.history['val_acc'])
                    self.history_train_loss.append(history.history['loss'])
                    self.history_vald_loss.append(history.history['val_loss'])
                    self.history_lr.append(history.history['lr'])

                    txtOut = self.print_history(history)
                    self.realtime_output(txtOut)

                    self.realtime_output("\n\n---------- END Batch {bnum} ---------- \n\n".format(bnum=b))
                self.realtime_output("********** END Tomo {tnum} ********** \n\n".format(tnum=t))
            self.realtime_output("########## END Epoch {epochnum} ########## \n".format(epochnum=e))
        self.save_history()

        end = time.clock()
        self.process_time = (end - start)

    def fetch_tomo(self):
        """
        this function reads only the current set of tomogram and its corresponding target
        """
        self.tomo = read_mrc(self.data.list_tomos_IDs[self.tomo_index])
        self.mask = read_mrc(self.data.list_masks_IDs[self.tomo_index])

        if self.tomo.shape != self.mask.shape:
            display("the tomogram and the target must be of the same size. " +
                    str(self.tomo.shape) + " is not equal to " + str(self.mask.shape) + ".")
            sys.exit()

        self.tomo = np.swapaxes(self.tomo, 0, 2)
        self.tomo = np.expand_dims(self.tomo, axis=0)
        self.tomo = np.expand_dims(self.tomo, axis=4)

        self.mask = np.swapaxes(self.mask, 0, 2)
        self.mask = np.expand_dims(self.mask, axis=0)
        self.mask = np.expand_dims(self.mask, axis=4)

        self.patches_tomo = tf.extract_volume_patches(self.tomo,
                                                      [1, self.obj.patch_size, self.obj.patch_size, 32, 1],
                                                      [1, self.obj.patch_size, self.obj.patch_size, 32, 1],
                                                      padding='VALID')
        self.patches_tomo = tf.reshape(self.patches_tomo, [-1, self.obj.patch_size, self.obj.patch_size, 32])
        self.patches_tomo = tf.squeeze(self.patches_tomo)

        self.patches_mask = tf.extract_volume_patches(self.mask,
                                                            [1, self.obj.patch_size, self.obj.patch_size, 32, 1],
                                                            [1, self.obj.patch_size, self.obj.patch_size, 32, 1],
                                                            padding='VALID')
        self.patches_mask = tf.reshape(self.patches_mask, [-1, self.obj.patch_size, self.obj.patch_size, 32])
        self.patches_mask = tf.squeeze(self.patches_mask)

        self.patches_tomo = self.patches_tomo.eval(session=tf.compat.v1.Session())
        self.patches_mask = self.patches_mask.eval(session=tf.compat.v1.Session())

        self.patches_tomo = np.swapaxes(self.patches_tomo, 1, 3)
        self.patches_mask = np.swapaxes(self.patches_mask, 1, 3)

    def fetch_batch(self):
        """
        this function fetches the patches from the current tomo based on the batch index
        """
        bstart = self.batch_idx * self.obj.batch_size
        bend = (self.batch_idx * self.obj.batch_size) + self.obj.batch_size

        self.batch_tomo = self.patches_tomo[bstart:bend]
        self.batch_mask = self.patches_mask[bstart:bend]

        # Patch base normalization
        self.batch_tomo = (self.batch_tomo - np.mean(self.batch_tomo)) / np.std(self.batch_tomo)
        self.batch_mask_onehot = []
        for m in range(len(self.batch_mask)):
            self.batch_mask_onehot.append(to_categorical(self.batch_mask[m], self.obj.classNum))
        # self.batch_mask_onehot = to_categorical(self.batch_mask, self.obj.classNum)

    def realtime_output(self, newstr):
        self.printstr = self.printstr + newstr
        self.obj.ui.textEdit.setText(self.printstr)

    def print_history(self, history):
        printstr = ""
        indxcol = 1
        for key, value in history.history.items():
            printstr = printstr + str(key) + " : " + str(value) + " | "
            if indxcol % 5 == 0:
                printstr = printstr + "\n"
            indxcol = indxcol + 1

        return printstr

    def save_history(self):
        # serialize model to JSON
        model_json = self.net.to_json()
        with open(os.path.join(self.data.output_path, "model.json"), "w") as json_file:
            json_file.write(model_json)

        save_csv(self.history_train_acc, self.data.output_path, "Train", "Accuracy_Details")
        save_csv(self.history_vald_acc, self.data.output_path, "Validation", "Accuracy_Details")
        save_csv(self.history_train_loss, self.data.output_path, "Train", "Loss_Details")
        save_csv(self.history_vald_loss, self.data.output_path, "Validation", "Loss_Details")
        save_csv(self.history_lr, self.data.output_path, "Train", "LearningRate_Details")

        # averaging the accuracy and loss over all folds
        self.train_acc = [np.mean([x[i] for x in self.history_train_acc]) for i in range(self.obj.epochs)]
        self.vald_acc = [np.mean([x[i] for x in self.history_vald_acc]) for i in range(self.obj.epochs)]
        self.train_loss = [np.mean([x[i] for x in self.history_train_loss]) for i in range(self.obj.epochs)]
        self.vald_loss = [np.mean([x[i] for x in self.history_vald_loss]) for i in range(self.obj.epochs)]
        self.train_lr = [np.mean([x[i] for x in self.history_lr]) for i in range(self.obj.epochs)]

        # saving the average results from folds
        save_csv(self.train_acc, self.data.output_path, flag="Train", name="Averaged_Accuracy")
        save_csv(self.vald_acc, self.data.output_path, flag="Validation", name="Averaged_Accuracy")
        save_csv(self.train_loss, self.data.output_path, flag="Train", name="Averaged_Loss")
        save_csv(self.vald_loss, self.data.output_path, flag="Validation", name="Averaged_Loss")
        save_csv(self.train_lr, self.data.output_path, flag="Train", name="Averaged_LearningRate")

    def plots(self):
        start_point = 10  # dropping the first few point in plots due to unstable behavior of model
        # cnf_matrix = np.zeros(shape=[self.obj.classNum, self.obj.classNum])

        plt.figure(num=1, figsize=(8, 6), dpi=100)
        plot_train_vs_vald(self.history_train_loss[start_point:], self.history_vald_loss[start_point:],
                           self.obj.output_path, self.obj.epochs, is_loss=True)

        plt.figure(num=2, figsize=(8, 6), dpi=100)
        plot_train_vs_vald(self.history_train_acc[start_point:], self.history_vald_acc[start_point:],
                           self.obj.output_path, self.obj.epochs)

        plt.figure(num=3, figsize=(8, 6), dpi=100)
        plot_lr(self.history_lr[start_point:], self.obj.output_path, self.obj.epochs)

        # Plot all ROC curves
        # plt.figure(num=4, figsize=(8, 6), dpi=100)
        # plot_roc(self.test_labels_one_hot_coded, self.test_predicted_probs,
        #          self.obj.classNum, self.obj.output_path, self.obj.epochs)

        # # Compute confusion matrix
        # cnf_matrix = confusion_matrix(self.test_labels, self.test_predicted_labels)
        # np.set_printoptions(precision=2)
        #
        # cnf_matrix2 = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        # print(np.average(cnf_matrix2.diagonal()))
        #
        # # Plot and save non-normalized confusion matrix
        # plt.figure(num=5, figsize=(5, 5), dpi=100)
        # plot_confusion_matrix(cnf_matrix, self.obj.classNum, self.obj.output_path, self.obj.epochs)
        #
        # # Plot normalized confusion matrix
        # plt.figure(num=6, figsize=(5, 5), dpi=100)
        # plot_confusion_matrix(cnf_matrix, self.obj.classNum, self.obj.output_path, self.obj.epochs, normalize=True)
        #
        # cnf_matrix2 = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        # print(np.average(cnf_matrix2.diagonal()))

    def save(self):

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

    def set_optimizer(self):
        self.optimizer = Adam(lr=self.lr, beta_1=.9, beta_2=.999, epsilon=1e-08, decay=0.0)

        if self.obj.opt == "SGD":
            self.optimizer = SGD(lr=self.lr, decay=0.0, momentum=0.9, nesterov=True)
        elif self.obj.opt == "Adagrad":
            self.obj.optimizer = Adagrad(lr=self.lr, epsilon=1e-08, decay=0.0)
        elif self.obj.opt == "Adadelta":
            self.optimizer = Adadelta(lr=self.lr, rho=0.95, epsilon=1e-08, decay=0.0)
        elif self.obj.opt == "Adamax":
            self.optimizer = Adamax(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        elif self.obj.opt == "Nadam":
            self.optimizer = Nadam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        elif self.obj.opt == "RMSprop":
            self.optimizer = RMSprop(lr=self.lr, rho=0.9, epsilon=1e-08, decay=0.0)

    # learning rate schedule
    def step_decay(self):
        initial_lrate = self.lr
        drop = 0.5
        epochs_drop = 50
        lrate = initial_lrate * math.pow(drop, math.floor((1 + self.obj.epochs) / epochs_drop))
        self.obj.lrr.append(lrate)
        return lrate

    def set_lr(self, lr_type):
        # learning schedule callback
        self.lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=10,
                                        min_lr=1e-06, mode='min', verbose=1)
        if lr_type == "step_decay":
            self.lr = LearningRateScheduler(self.step_decay())
        elif lr_type == "lambda":
            self.lr = LearningRateScheduler(lambda this_epoch: self.obj.lr * 0.99 ** this_epoch)
        elif lr_type == "cyclic":
            self.lr = CyclicLR(base_lr=self.obj.lr, max_lr=6e-04, step_size=500., mode='exp_range', gamma=0.99994)

    def get_lr_metric(self):
        def lr(y_true, y_pred):
            # lrr.append(float(K.get_value(optimizer.lr)))
            self.lr = self.optimizer.lr

    def set_compile(self):
        # self.get_lr_metric()

        # self.obj.metrics = ['accuracy', self.lr]
        self.obj.metrics = [keras.metrics.TruePositives(name='tp'),
                            keras.metrics.FalsePositives(name='fp'),
                            keras.metrics.TrueNegatives(name='tn'),
                            keras.metrics.FalseNegatives(name='fn'),
                            keras.metrics.BinaryAccuracy(name='acc'),
                            keras.metrics.Precision(name='precision'),
                            keras.metrics.Recall(name='recall'),
                            keras.metrics.AUC(name='auc')
                            ]
        if self.obj.loss != "tversky":
            self.net.compile(optimizer=self.optimizer, loss=self.obj.loss, metrics=[self.obj.metrics])
        else:
            self.net.compile(optimizer=self.optimizer, loss=self.tversky_loss, metrics=[self.obj.metrics])

    def set_checkpoint(self):
        checkpoint_dir = os.path.join(self.data.output_path,
                                      'weights-improvement-{epoch:03d}-{acc:.2f}-{loss:.2f}-{val_acc:.2f}-{val_loss:.2f}.hdf5')
        self.checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    def set_callback(self):
        # checkpoint results
        self.set_checkpoint()

        # set learning schedule
        self.set_lr(lr_type="reduce_lr")

        self.callbacks = [self.checkpoint, self.lr]

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
        setting_info = setting_info + "\nLearning Rate = " + str(self.lr)
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

    def tversky_loss(self, y_true, y_pred):
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
        self.net = Model(input_img, outputs)

    def unet3d(self):
        # The UNET model from DeepFinder
        input_img = layers.Input(shape=(32, self.obj.patch_size, self.obj.patch_size, 1))

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

        self.net = Model(input_img, output)
