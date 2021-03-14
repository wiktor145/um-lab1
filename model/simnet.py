import keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

INPUT_SHAPE = (1, 512, 2)
DELTA = 0.6


def simnet():
    inputs = layers.Input(shape=INPUT_SHAPE)
    x = layers.LayerNormalization(axis=-1)(inputs)
    x = layers.Conv2D(4096, (1, 512), strides=(1, 1), use_bias=True, data_format="channels_last")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(4096, (1, 1), strides=(1, 1), use_bias=True)(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(1, (1, 1), strides=(1, 1), use_bias=True)(x)
    x = layers.Activation("sigmoid")(x)

    model = models.Model(inputs, x, name="simnet")

    def loss(y_true, y_pred):
        def cosine_similarity(x1, x2):
            dot_product = K.sum(x1 * x2, axis=-1)
            x1_len = K.sum(x1 * x1, axis=-1)
            x2_len = K.sum(x2 * x2, axis=-1)
            return dot_product / (x1_len * x2_len)

        # x = K.l2_normalize(x1, axis=-1)
        # y = K.l2_normalize(x2, axis=-1)
        # return K.sum(x * y, axis=-1, keepdims=True)

        x1 = model.inputs[0][0, :, 0]
        x2 = model.inputs[0][0, :, 1]
        l = 1.0 if y_true == 1 else 0.0
        sim = cosine_similarity(x1, x2)
        return abs(y_pred - l * (sim + DELTA) - (1 - l) * (sim - DELTA))

    #optimizer = Adam(lr=0.001)

    model.compile(optimizer="adam", loss="binary_crossentropy")
    #model.summary()

    return model
