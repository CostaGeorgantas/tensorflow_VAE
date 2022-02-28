import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Conv2DTranspose, Reshape


class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_layers = Sequential(
            [
                Conv2D(32, 3, activation='relu'), # 28
                BatchNormalization(),
                Conv2D(32, 3, activation='relu', strides=2, padding="same"), # 13
                Conv2D(64, 3, activation='relu', strides=2, padding="same"), # 7
                Conv2D(128, 3, activation="relu", strides=2, padding="same"), # 4
                Flatten()
            ]
        )
        self.mu_layer = Dense(2)
        self.sig_layer = Dense(2)

    def call(self, x):
        x = self.encoder_layers(x)
        mu, sig = self.mu_layer(x), self.sig_layer(x)
        return mu, sig


class Decoder(Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_layers = Sequential(
            [
            Dense(49),
            Reshape((7, 7, 1)),
            Conv2DTranspose(16, 3, strides=2),
            Conv2DTranspose(8, 3, strides=2),
            Conv2D(1, 4)

        ])

    def call(self, mu, var):
        var = tf.math.exp(var * 0.5)
        x = tf.random.normal(tf.shape(mu)) * var + mu # sample
        x = self.decoder_layers(x)
        return x


def vae_loss(x, x_hat, mu, var):
    kl_loss = 0.5 * (- tf.math.log(var) + tf.math.pow(mu, 2) + var - 1)
    kl_loss = tf.math.reduce_sum(kl_loss)
    recon_loss = tf.keras.metrics.binary_crossentropy(x, x_hat)
    recon_loss = tf.math.reduce_sum(recon_loss)
    return kl_loss + recon_loss


class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, x):
        with tf.GradientTape() as tape:
            mu, logvar = self.encoder(x)
            var = tf.exp(logvar * .5)
            x_hat = self.decoder(mu, var)
            loss = vae_loss(x, x_hat, mu, var)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {"loss": loss}
