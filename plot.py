import os
import numpy as np
import tensorflow as tf
from model import Encoder, Decoder, VAE
import matplotlib.pyplot as plt


def get_latent(model, x):
    mu, _ = model.encoder(x)
    return mu

def plot_latent(model, x, labels):
    mu, _ = model.encoder(x)
    plt.scatter(mu[:, 0], mu[:, 1], c=labels)
    #plt.colorbar()
    plt.axis("off")
    plt.show()


def plot_latent_examples(model, n=20, figsize=20):
    #shamelessly copied from https://keras.io/examples/generative/vae/
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 2.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            #z_sample = np.array([[xi, yi] + 8 * [0.0]])
            z_sample = np.array([[xi, yi]])
            x_decoded = model.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))

    plt.axis("off")
    color_map = plt.cm.get_cmap('Greys_r')
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap=color_map.reversed())
    plt.show()


def plot_image_from_latent(model, z_sample):

    x_decoded = model.decoder.predict(z_sample)
    plt.axis("off")
    plt.imshow(x_decoded.reshape(28, 28), cmap="Greys_r")
    plt.show()


def plot_orig_and_reconstr(model, x):
    idx=4000
    mu = get_latent(model, x)
    img = x[idx].reshape(28, 28)
    plt.axis("off")
    plt.imshow(img.reshape(28, 28), cmap="Greys_r")
    plt.show()

    plot_image_from_latent(model, mu[idx].numpy().reshape(1, -1))


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    checkpoint_path = "checkpoints/model_nosample"

    model = tf.keras.models.load_model(checkpoint_path)

    plot_latent(model, x_train, y_train)
    plot_latent_examples(model)
