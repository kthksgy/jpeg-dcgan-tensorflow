import tensorflow as tf
from tensorflow.keras.optimizers import (
    SGD, Adam
)

__CROSSENTROPY = tf.keras.losses.BinaryCrossentropy(from_logits=True)
__GENERATOR_OPTIMIZER = Adam(lr=0.0002, beta_1=0.5)
__DISCRIMINATOR_OPTIMIZER = SGD(lr=0.01)


def __discriminator_loss(real_output, fake_output):
    real_loss = __CROSSENTROPY(tf.ones_like(real_output), real_output)
    fake_loss = __CROSSENTROPY(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def __generator_loss(fake_output):
    return __CROSSENTROPY(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(real_images, z_dim: int, generator, discriminator):
    noise = tf.random.normal([real_images.shape[0], z_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = __generator_loss(fake_output)
        dis_loss = __discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        dis_loss, discriminator.trainable_variables)

    __GENERATOR_OPTIMIZER.apply_gradients(zip(
        gradients_of_generator, generator.trainable_variables))
    __DISCRIMINATOR_OPTIMIZER.apply_gradients(zip(
        gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, dis_loss
