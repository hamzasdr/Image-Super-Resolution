from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda,UpSampling2D,Dense
from tensorflow.python.keras.models import Model, Sequential
import tensorflow as tf

def edsr(scale = 2, filters =64, res_blocks=8):
    x_in = Input(shape=(128, 128, 3))

    x = b = Conv2D(filters, 3, padding='same')(x_in)
    for i in range(res_blocks):
        b = res_block(b, filters)
    b = Conv2D(filters, 3, padding='same')(b)
    x = Add()([x, b])
    x = upsample(x, scale, filters)
    x = Conv2D(3, 3, padding='same')(x)

    return Model(x_in, x, name="edsr")


def res_block(x_in, filters):
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = Conv2D(filters, 3, padding='same')(x)
    x = Add()([x_in, x])
    return x


def upsample(x, scale, num_filters):
    if scale == 2:
        x = Conv2D(num_filters * (2 ** 2), 3, padding='same', name='scale_2')(x)
        x = Lambda(PS(scale=2))(x)
    #TODO : implement the other scaling options
    return x

def PS(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)
