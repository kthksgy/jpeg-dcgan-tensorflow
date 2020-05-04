from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Reshape, Dense, Flatten,
    Conv2D, Conv2DTranspose,
    BatchNormalization, LeakyReLU, Dropout)


def make_generator(z_dim: int):
    inputs = Input((z_dim,), name='generator_input')
    x = inputs

    x = Dense(
        2 * 2 * z_dim * 4, use_bias=False,
        name='dense1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = LeakyReLU(name='act1')(x)
    x = Reshape((2, 2, z_dim * 4), name='reshape1')(x)

    x = Conv2DTranspose(
        z_dim * 2, 3, padding='same', use_bias=False,
        name='deconv4')(x)
    x = BatchNormalization(name='bn4')(x)
    x = LeakyReLU(name='act4')(x)

    x = Conv2DTranspose(
        z_dim, 3, padding='same', use_bias=False,
        name='deconv5')(x)
    x = BatchNormalization(name='bn5')(x)
    x = LeakyReLU(name='act5')(x)

    x = Conv2DTranspose(
        64, 3, strides=2, padding='same', use_bias=False,
        name='generator_output')(x)

    model = Model(inputs, x, name='mnist_dct_generator')
    return model


def make_discriminator(z_dim: int):
    inputs = Input((4, 4, 64), name='generator_input')
    x = inputs

    x = Conv2D(
        z_dim // 2, 3, padding='same',
        name='conv1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = LeakyReLU(name='act1')(x)
    x = Dropout(0.3, name='drop1')(x)

    x = Conv2D(
        z_dim, 3, padding='same',
        name='conv3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = LeakyReLU(name='act3')(x)
    x = Dropout(0.3, name='drop3')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(1, name='discriminator_output')(x)
    model = Model(inputs, x, name='mnist_dct_discriminator')
    return model


if __name__ == '__main__':
    generator = make_generator(100)
    generator.summary()

    discriminator = make_discriminator(100)
    discriminator.summary()
