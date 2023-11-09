import keras
import tensorflow as tf
from keras.layers import Conv2D
from keras import backend as K


class RegionalProposalNetwork(keras.Model):
    def __init__(self, max_proposals_pre_nms_train, max_proposals_post_nms_train, max_proposals_pre_nms_infer,
                 max_proposals_post_nms_infer, l2=0, allow_edge_proposals=False):
        super().__init__()

        self._max_proposals_pre_nms_train = max_proposals_pre_nms_train
        self._max_proposals_post_nms_train = max_proposals_post_nms_train
        self._max_proposals_pre_nms_infer = max_proposals_pre_nms_infer
        self._max_proposals_post_nms_infer = max_proposals_post_nms_infer
        self._allow_edge_proposals = allow_edge_proposals

        regularizer = tf.keras.regularizers.l2(l2)
        initial_weights = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

        anchors_per_location = 9

        self._rpn_conv1 = Conv2D(name="rpn_conv1", kernel_size=(3, 3), strides=1, filters=512, padding="same", activation="relu", kernel_initializer=initial_weights, kernel_regularizer=regularizer)
        self._rpn_class = Conv2D(name="rpn_class", kernel_size=(1, 1), strides=1, filters=anchors_per_location, padding="same", activation="sigmoid", kernel_initializer=initial_weights)
        self._rpn_boxes = Conv2D(name="rpn_boxes", kernel_size=(1, 1), strides=1, filters=4 * anchors_per_location, padding="same", activation=None, kernel_initializer=initial_weights)

    def call(self, _input, training):
        pass