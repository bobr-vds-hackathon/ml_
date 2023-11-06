import tensorflow as tf
from tensorflow import keras
from keras import models
from keras.layers import Dense, Dropout, Flatten, Lambda, TimeDistributed
from roi_pooling_layer import RoIPoolingLayer


class Detector(keras.models.Model):
    def __init__(self, num_detection_classes: int, custom_roi_pool, activate_class_outputs, l2, dropout_probability):
        super().__init__()

        self._num_classes = num_detection_classes
        self._activate_class_outputs = activate_class_outputs
        self._dropout_probability = dropout_probability

        regularization = tf.keras.regularizers.l2(l2)
        class_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        regressor_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001)

        self._roi_pooling_layer = RoIPoolingLayer(pool_size=7) if custom_roi_pool else None

        self._flatten_layer = TimeDistributed(Flatten())
        self._flatten_convolution = TimeDistributed(layer=Dense(units=4096, activation='relu', kernel_regularizer=regularization))
        self._dropout_layer = TimeDistributed(Dropout(dropout_probability))
        self._flatten_convolution2 = TimeDistributed(layer=Dense(units=4096, activation='relu', kernel_regularizer=regularization))
        self._dropout_layer2 = TimeDistributed(Dropout(dropout_probability))

        subclass_activation = "softmax"
        self._classifier = TimeDistributed(layer=Dense(units=num_detection_classes, activation= subclass_activation, kernel_initializer=class_initializer))

        self._regressor = TimeDistributed(layer=Dense(units= 4 * (num_detection_classes - 1), activation= "linear", kernel_initializer=regressor_initializer))

    def call(self, input_, training):
        pass    