import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Reshape, Dense, Flatten,
    Conv2D, Conv2DTranspose,
    BatchNormalization, LeakyReLU, Dropout)


def make_generator(z_dim: int):
    inputs = Input((z_dim,), name='generator_input')
    x = inputs

    x = Dense(
        4 * 4 * 512, use_bias=True,
        name='g_dense1')(x)
    x = BatchNormalization(name='g_bn1')(x)
    x = LeakyReLU(name='g_act1')(x)
    x = Reshape((4, 4, 512), name='g_reshape1')(x)

    x = Conv2DTranspose(
        256, 3, padding='same', use_bias=True,
        name='g_deconv2')(x)
    x = BatchNormalization(name='g_bn2')(x)
    x = LeakyReLU(name='g_act2')(x)

    # x = Conv2DTranspose(
    #     256, 3, padding='same', use_bias=True,
    #     name='g_deconv3')(x)
    # x = BatchNormalization(name='g_bn3')(x)
    # x = LeakyReLU(name='g_act3')(x)

    x = Conv2DTranspose(
        128, 3, padding='same', use_bias=True,
        name='g_deconv4')(x)
    x = BatchNormalization(name='g_bn4')(x)
    x = LeakyReLU(name='g_act4')(x)

    # x = Conv2DTranspose(
    #     128, 3, padding='same', use_bias=True,
    #     name='g_deconv5')(x)
    # x = BatchNormalization(name='g_bn5')(x)
    # x = LeakyReLU(name='g_act5')(x)

    x = Conv2DTranspose(
        64, 3, strides=1, padding='same', use_bias=True,
        name='generator_output')(x)

    x *= 25

    model = Model(inputs, x, name='mnist_dct_generator')
    return model


def make_discriminator(z_dim: int):
    inputs = Input((4, 4, 64), name='generator_input')
    x = inputs

    # x = Conv2D(
    #     64, 3, padding='same',
    #     name='d_conv1')(x)
    # x = BatchNormalization(name='d_bn1')(x)
    # x = LeakyReLU(name='d_act1')(x)
    # x = Dropout(0.3, name='d_drop1')(x)

    x = Conv2D(
        128, 3, padding='same',
        name='d_conv2')(x)
    x = BatchNormalization(name='d_bn2')(x)
    x = LeakyReLU(name='d_act2')(x)
    x = Dropout(0.3, name='d_drop2')(x)

    x = Conv2D(
        256, 3, padding='same',
        name='d_conv3')(x)
    x = BatchNormalization(name='d_bn3')(x)
    x = LeakyReLU(name='d_act3')(x)
    x = Dropout(0.3, name='d_drop3')(x)

    x = Flatten(name='d_flatten')(x)
    x = Dense(1, name='discriminator_output')(x)
    model = Model(inputs, x, name='mnist_dct_discriminator')
    return model


if __name__ == '__main__':
    generator = make_generator(100)
    generator.summary()

    discriminator = make_discriminator(100)
    discriminator.summary()
