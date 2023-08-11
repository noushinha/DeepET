import tensorflow as tf
import keras.backend as bk
from keras.losses import binary_crossentropy, MeanSquaredError, MeanAbsoluteError, Huber
import numpy as np


beta = 0.25
alpha = 0.25
gamma = 2
epsilon = 1e-5
smooth = 1


class Semantic_loss_functions(object):
    def __init__(self):
        print ("semantic loss functions initialized")

    def dice_coef(self, y_true, y_pred):
        y_true_f = bk.flatten(y_true)
        y_pred_f = bk.flatten(y_pred)
        intersection = bk.sum(y_true_f * y_pred_f)
        return (2. * intersection + bk.epsilon()) / (
                    bk.sum(y_true_f) + bk.sum(y_pred_f) + bk.epsilon())

    def sensitivity(self, y_true, y_pred):
        true_positives = bk.sum(bk.round(bk.clip(y_true * y_pred, 0, 1)))
        possible_positives = bk.sum(bk.round(bk.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + bk.epsilon())

    def specificity(self, y_true, y_pred):
        true_negatives = bk.sum(
            bk.round(bk.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        possible_negatives = bk.sum(bk.round(bk.clip(1 - y_true, 0, 1)))
        return true_negatives / (possible_negatives + bk.epsilon())

    def convert_to_logits(self, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                                  1 - tf.keras.backend.epsilon())
        return tf.math.log(y_pred / (1 - y_pred))

    def weighted_cross_entropyloss(self, y_true, y_pred):
        y_pred = self.convert_to_logits(y_pred)
        pos_weight = beta / (1 - beta)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred,
                                                        targets=y_true,
                                                        pos_weight=pos_weight)
        return tf.reduce_mean(loss)

    def focal_loss_with_logits(self, logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(
            -logits)) * (weight_a + weight_b) + logits * weight_b

    def focal_loss(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                                  1 - tf.keras.backend.epsilon())
        logits = tf.math.log(y_pred / (1 - y_pred))

        loss = self.focal_loss_with_logits(logits=logits, targets=y_true,
                                      alpha=alpha, gamma=gamma, y_pred=y_pred)

        return tf.reduce_mean(loss)

    def depth_softmax(self, matrix):
        sigmoid = lambda x: 1 / (1 + bk.exp(-x))
        sigmoided_matrix = sigmoid(matrix)
        softmax_matrix = sigmoided_matrix / bk.sum(sigmoided_matrix, axis=0)
        return softmax_matrix

    def generalized_dice_coefficient(self, y_true, y_pred):
        smooth = 1.
        y_true_f = bk.flatten(y_true)
        y_pred_f = bk.flatten(y_pred)
        intersection = bk.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (
                    bk.sum(y_true_f) + bk.sum(y_pred_f) + smooth)
        return score

    def dice_loss(self, y_true, y_pred):
        loss = 1 - self.generalized_dice_coefficient(y_true, y_pred)
        return loss

    def bce_dice_loss(self, y_true, y_pred):
        loss = binary_crossentropy(y_true, y_pred) + \
               self.dice_loss(y_true, y_pred)
        return loss / 2.0

    def confusion(self, y_true, y_pred):
        smooth = 1
        y_pred_pos = bk.clip(y_pred, 0, 1)
        y_pred_neg = 1 - y_pred_pos
        y_pos = bk.clip(y_true, 0, 1)
        y_neg = 1 - y_pos
        tp = bk.sum(y_pos * y_pred_pos)
        fp = bk.sum(y_neg * y_pred_pos)
        fn = bk.sum(y_pos * y_pred_neg)
        prec = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)
        return prec, recall

    def true_positive(self, y_true, y_pred):
        smooth = 1
        y_pred_pos = bk.round(bk.clip(y_pred, 0, 1))
        y_pos = bk.round(bk.clip(y_true, 0, 1))
        tp = (bk.sum(y_pos * y_pred_pos) + smooth) / (bk.sum(y_pos) + smooth)
        return tp

    def true_negative(self, y_true, y_pred):
        smooth = 1
        y_pred_pos = bk.round(bk.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos
        y_pos = bk.round(bk.clip(y_true, 0, 1))
        y_neg = 1 - y_pos
        tn = (bk.sum(y_neg * y_pred_neg) + smooth) / (bk.sum(y_neg) + smooth)
        return tn

    def tversky_index(self, y_true, y_pred):
        y_true_pos = bk.flatten(y_true)
        y_pred_pos = bk.flatten(y_pred)
        true_pos = bk.sum(y_true_pos * y_pred_pos)
        false_neg = bk.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = bk.sum((1 - y_true_pos) * y_pred_pos)
        alpha = 0.5
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (
                    1 - alpha) * false_pos + smooth)

    def tversky_loss(self, y_true, y_pred):
        return 1 - self.tversky_index(y_true, y_pred)

    def focal_tversky(self, y_true, y_pred):
        pt_1 = self.tversky_index(y_true, y_pred)
        gamma = 0.75
        return bk.pow((1 - pt_1), gamma)

    def log_cosh_dice_loss(self, y_true, y_pred):
        x = self.dice_loss(y_true, y_pred)
        return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)

    def weighted_log_loss(self, y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= bk.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = bk.clip(y_pred, bk.epsilon(), 1 - bk.epsilon())
        # weights are assigned in this order : normal,necrotic,edema,enhancing
        weights = np.array([0.001361341, 0.37459406, 0.61179253])
        loss = y_true * bk.log(y_pred) * weights
        loss = bk.mean(-bk.sum(loss, -1))
        return loss

    def mse(self, y_true, y_pred):
        mse = MeanSquaredError(reduction="auto")
        return mse(y_true, y_pred).numpy()


    def mae(self, y_true, y_pred):
        mae = MeanAbsoluteError(reduction="auto")
        return mae(y_true, y_pred).numpy()

    def huber(self, y_true, y_pred):
        huber= Huber(reduction="auto", delta=1.0)
        return huber(y_true, y_pred).numpy()

    # def gen_dice_loss(self, y_true, y_pred):
    #     '''
    #     computes the sum of two losses : generalised dice loss and weighted cross entropy
    #     '''
    #
    #     # generalised dice score is calculated as in this paper : https://arxiv.org/pdf/1707.03237
    #     y_true_f = bk.reshape(y_true, shape=(-1, 4))
    #     y_pred_f = bk.reshape(y_pred, shape=(-1, 4))
    #     sum_p = bk.sum(y_pred_f, axis=-2)
    #     sum_r = bk.sum(y_true_f, axis=-2)
    #     sum_pr = bk.sum(y_true_f * y_pred_f, axis=-2)
    #     weights = bk.pow(bk.square(sum_r) + bk.epsilon(), -1)
    #     generalised_dice_numerator = 2 * bk.sum(weights * sum_pr)
    #     generalised_dice_denominator = bk.sum(weights * (sum_r + sum_p))
    #     generalised_dice_score = generalised_dice_numerator / generalised_dice_denominator
    #     GDL = 1 - generalised_dice_score
    #     del sum_p, sum_r, sum_pr, weights
    #
    #     return GDL + self.weighted_log_loss(y_true, y_pred)

    # def Combo_loss(self, targets, inputs, eps=1e-9):
    #     ALPHA = 0.5  # < 0.5 penalises FP more, > 0.5 penalises FN more
    #     CE_RATIO = 0.5  # weighted contribution of modified CE loss compared to Dice loss
    #
    #     targets = bk.flatten(targets)
    #     inputs = bk.flatten(inputs)
    #
    #     intersection = bk.sum(targets * inputs)
    #     dice = (2. * intersection + smooth) / (bk.sum(targets) + bk.sum(inputs) + smooth)
    #     inputs = bk.clip(inputs, eps, 1.0 - eps)
    #     out = - (ALPHA * ((targets * bk.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * bk.log(1.0 - inputs))))
    #     weighted_ce = bk.mean(out, axis=-1)
    #     combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
    #
    #     return combo


