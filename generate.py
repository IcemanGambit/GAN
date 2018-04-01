
# coding: utf-8

# In[ ]:


from __future__ import print_function

from keras import backend as K
K.set_image_dim_ordering('th')

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD, Adam
from keras.datasets import mnist
from keras import utils
import numpy as np
from PIL import Image, ImageOps
import argparse
import math

import os
import os.path

import glob


# In[ ]:


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*8*8))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((128, 8, 8), input_shape=(128*8*8,)))
    model.add(UpSampling2D(size=(4, 4)))
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(4, 4)))
    model.add(Convolution2D(1, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    return model

# In[ ]:


model = generator_model()
print(model.summary())


# In[ ]:

def clean(image):
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if image[i][j] + image[i+1][j] + image[i][j+1] + image[i-1][j] + image[i][j-1] > 127 * 5:
                image[i][j] = 255
    return image

def generate(BATCH_SIZE):
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('goodgenerator.h5')
    noise = np.zeros((BATCH_SIZE, 100))
    a = np.random.uniform(-1, 1, 100)
    b = np.random.uniform(-1, 1, 100)
    grad = (b - a) / BATCH_SIZE
    for i in range(BATCH_SIZE):
        noise[i, :] = np.random.uniform(-1, 1, 100)
    generated_images = generator.predict(noise, verbose=1)
    print(generated_images.shape)
    i=1
    for image in generated_images:
        image = image[0]
        image = image*127.5+127.5
        Image.fromarray(image.astype(np.uint8)).save("original"+str(i)+".png")
        #Image.fromarray(image.astype(np.uint8)).show()
        clean(image)
        image = Image.fromarray(image.astype(np.uint8))
        #image.show()        
        image.save("sharpen"+str(i)+".png")
        i=i+1


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args


# In[ ]:


generate(6)

