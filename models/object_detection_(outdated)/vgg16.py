from tensorflow import keras
from keras.initializers import glorot_normal
from keras.layers import Conv2D, MaxPooling2D


class FeatureExtractor(keras.Model):
    def __init__(self, l2=0):
        super().__init__()

        initial_weights = glorot_normal()
        regularization = keras.regularizers.l2(l2)

        input_shape = (None, None, 3)

        # первый индекс показывает принадлежность к блоку, второй порядковый номер
        self._1_conv_1 = Conv2D(name="_1_conv_1", input_shape=input_shape, kernel_size=(3, 3), strides=1, filters=64, padding="same", activation="relu", kernel_initializer=initial_weights, trainable=False)
        self._1_conv_2 = Conv2D(name="_1_conv_2", kernel_size=(3, 3), strides=1, filters=64, padding="same", activation="relu", kernel_initializer=initial_weights, trainable=False)
        self._1_maxpool = MaxPooling2D(pool_size=2, strides=2)

        self._2_conv_1 = Conv2D(name="_2_conv_1", kernel_size=(3, 3), strides=1, filters=128, padding="same", activation="relu", kernel_initializer=initial_weights, trainable=False)
        self._2_conv_1 = Conv2D(name="_1_conv_2", kernel_size=(3, 3), strides=1, filters=128, padding="same", activation="relu", kernel_initializer=initial_weights, trainable=False)
        self._2_maxpool = MaxPooling2D(pool_size=2, strides=2)

        self._3_conv1 = Conv2D(name="_3_conv1", kernel_size=(3, 3), strides=1, filters=256, padding="same", activation="relu", kernel_initializer=initial_weights, kernel_regularizer=regularization)
        self._3_conv2 = Conv2D(name="_3_conv2", kernel_size=(3, 3), strides=1, filters=256, padding="same", activation="relu", kernel_initializer=initial_weights, kernel_regularizer=regularization)
        self._3_conv3 = Conv2D(name="_3_conv3", kernel_size=(3, 3), strides=1, filters=256, padding="same", activation="relu", kernel_initializer=initial_weights, kernel_regularizer=regularization)
        self._3_maxpool = MaxPooling2D(pool_size=2, strides=2)

        self._4_conv1 = Conv2D(name="_4_conv1", kernel_size=(3, 3), strides=1, filters=512, padding="same", activation="relu", kernel_initializer=initial_weights, kernel_regularizer=regularization)
        self._4_conv2 = Conv2D(name="_4_conv2", kernel_size=(3, 3), strides=1, filters=512, padding="same", activation="relu", kernel_initializer=initial_weights, kernel_regularizer=regularization)
        self._4_conv3 = Conv2D(name="_4_conv3", kernel_size=(3, 3), strides=1, filters=512, padding="same", activation="relu", kernel_initializer=initial_weights, kernel_regularizer=regularization)
        self._4_maxpool = MaxPooling2D(pool_size=2, strides=2)

        self._5_conv1 = Conv2D(name="block5_conv1", kernel_size=(3, 3), strides=1, filters=512, padding="same", activation="relu", kernel_initializer=initial_weights, kernel_regularizer=regularization)
        self._5_conv2 = Conv2D(name="_5_conv2", kernel_size=(3, 3), strides=1, filters=512, padding="same", activation="relu", kernel_initializer=initial_weights, kernel_regularizer=regularization)
        self._5_conv3 = Conv2D(name="_5_conv3", kernel_size=(3, 3), strides=1, filters=512, padding="same", activation="relu", kernel_initializer=initial_weights, kernel_regularizer=regularization)
