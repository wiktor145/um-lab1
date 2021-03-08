import keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from tensorflow_addons.optimizers import SGDW

INPUT_SHAPE = (1, 512, 2)
DELTA = 0.6


def simnet():
    inputs = layers.Input(shape=INPUT_SHAPE)
    x = layers.Conv2D(4096, (1, 512), strides=(1, 1), use_bias=True, data_format="channels_last")(inputs)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(4096, (1, 1), strides=(1, 1), use_bias=True)(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(1, (1, 1), strides=(1, 1), use_bias=True)(x)

    model = models.Model(inputs, x, name="simnet")

    def loss(y_true, y_pred):
        def cosine_similarity(x1, x2):
            dot_product = K.sum(x1 * x2, axis=-1)
            x1_len = K.sum(x1 * x1, axis=-1)
            x2_len = K.sum(x2 * x2, axis=-1)
            return dot_product / (x1_len * x2_len)

        x1 = model.inputs[0][0, :, 0]
        x2 = model.inputs[0][0, :, 1]
        l = 1.0 if y_true == 1 else 0.0
        sim = cosine_similarity(x1, x2)
        return abs(y_pred - l * (sim + DELTA) - (1 - l) * (sim - DELTA))

    optimizer = SGDW(learning_rate=0.001, momentum=0.9, weight_decay=0.0005)

    model.compile(optimizer=optimizer, loss=loss)

    return model
