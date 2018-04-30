from __future__ import print_function, division

from keras.datasets import mnist
from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import tensorflow as tf
from scipy.misc import imread, imsave

import matplotlib.pyplot as plt

import sys
import os
from PIL import Image
from glob import glob

import numpy as np

class GAN():
    def __init__(self):
        self.img_rows = 32 
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)

        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        noise_shape = (100,)
        
        model = Sequential()

        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)


    def generate(self, batch_size, mode='L', weights=True):
        
        if weights:
            self.generator.load_weights('goodgenerator.h5')

        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = (1/2.5) * gen_imgs + 0.5

        fig, axs = plt.subplots(1,1)

        for i in range(batch_size):
            if mode=='L':
                axs.imshow(gen_imgs[i, :,:,0], cmap='gray')
            else:
                axs.imshow(gen_imgs[i, :,:,:])
            axs.axis('off')
            fig.set_dpi(15)
            fig.savefig("generated/%d.png" % i, bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.generate(batch_size=15, mode='RGB', weights=True)