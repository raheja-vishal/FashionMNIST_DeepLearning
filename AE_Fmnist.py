import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from scipy.stats import norm
import keras
from keras import layers
from keras.models import Model
from keras import metrics
from keras import backend as K
import tensorflow as tf
(x_train, y_train), (x_test, y_test)=tf.keras.datasets.fashion_mnist.load_data()

#encoded representations
encoding_dim = 2  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

#input image
input_img = keras.Input(shape=(784,))
# encoded representation of the input
encoded = layers.Dense(32, activation='relu')(input_img)
encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

#lossy reconstruction of the input
decoded = layers.Dense(32, activation='relu')(encoded)
decoded = layers.Dense(784, activation='sigmoid')(decoded)

#reconstruction
autoencoder = keras.Model(input_img, decoded)


# encoded representation
encoder = keras.Model(input_img, encoded)

# encoded input
encoded_input = keras.Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-2]
#decoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
autoencoder.save("AE.h5")

plt.figure(figsize=(10, 10))
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test, cmap='brg')
plt.colorbar()
# plt.show()
plt.savefig('UNTRAINED_CLUSTER.png')
for idx in range(1,2):

    autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=512,
                shuffle=True,
                verbose=2,
                validation_data=(x_test, x_test))

    autoencoder.save("AE.h5")
    encoded_imgs = encoder.predict(x_test)
    # decoded_imgs = decoder.predict(encoded_imgs)

    plt.figure(figsize=(10, 10))
    plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test, cmap='brg')
    plt.colorbar()
    # plt.show()
    plt.savefig('TRAINED_CLUSTER_epoch_{}.png'.format(idx))


