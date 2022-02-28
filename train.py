import tensorflow as tf
from model import Encoder, Decoder, VAE

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    optimizer = tf.keras.optimizers.Adam(1e-4)
    enc = Encoder()
    dec = Decoder()
    model = VAE(enc, dec)
    model.compile(optimizer=optimizer)

    model.fit(x_train, epochs=100, batch_size=128)
