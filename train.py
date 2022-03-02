import tensorflow as tf

from model import Encoder, Decoder, VAE

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    checkpoint_path = "checkpoints/model_noreg"

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, monitor='loss') # saving only weights does not work when loading back

    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    optimizer = tf.keras.optimizers.Adam(1e-4)
    enc = Encoder()
    dec = Decoder()
    inputs = tf.keras.Input(shape=(None, 28, 28, 1))
    model = VAE(enc, dec)
    model.compute_output_shape((None, 28, 28, 1)) # model.build does NOT work ???
    model.compile(optimizer=optimizer)


    model.fit(x_train, epochs=50, batch_size=128, callbacks=[cp_callback])
