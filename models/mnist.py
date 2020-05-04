from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Reshape, Dense, Flatten,
    Conv2D, Conv2DTranspose,
    BatchNormalization, LeakyReLU, Dropout)


def make_generator(z_dim: int):
    inputs = Input((z_dim,), name='generator_input')
    x = inputs

    x = Dense(
        7 * 7 * z_dim * 2, use_bias=False,
        name='dense1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = LeakyReLU(name='act1')(x)
    x = Reshape((7, 7, z_dim * 2), name='reshape1')(x)

    x = Conv2DTranspose(
        z_dim, 5, padding='same', use_bias=False,
        name='deconv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = LeakyReLU(name='act2')(x)

    x = Conv2DTranspose(
        z_dim // 2, 5, strides=2, padding='same', use_bias=False,
        name='deconv3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = LeakyReLU(name='act3')(x)

    x = Conv2DTranspose(
        1, 5, strides=2, padding='same', activation='tanh', use_bias=False,
        name='generator_output')(x)

    model = Model(inputs, x, name='mnist_generator')
    return model


def make_discriminator(z_dim: int):
    inputs = Input((28, 28, 1), name='generator_input')
    x = inputs

    x = Conv2D(
        z_dim // 2, 5, strides=2, padding='same',
        name='conv1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = LeakyReLU(name='act1')(x)
    x = Dropout(0.3, name='drop1')(x)

    x = Conv2D(
        z_dim, 5, strides=2, padding='same',
        name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = LeakyReLU(name='act2')(x)
    x = Dropout(0.3, name='drop2')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(1, name='discriminator_output')(x)
    model = Model(inputs, x, name='mnist_discriminator')
    return model


if __name__ == '__main__':
    generator = make_generator(100)
    generator.summary()

    discriminator = make_discriminator(100)
    discriminator.summary()
