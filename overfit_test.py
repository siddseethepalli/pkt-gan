#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import math
import os
import sys

from six.moves import range
import keras.backend as K
from keras.layers import Activation, Dense, Embedding, Flatten, Input, multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import SGD
import numpy as np
import pylab

K.set_image_dim_ordering('th')

COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ALPHA = 100
NUM_CLUSTERS = 3
NUM_DISCRIMINATORS = 1

def discriminator_loss(target, output):
    return -K.mean(target*output)

def gaussian_mixture_circle(batchsize, num_cluster=3, scale=3, std=0.5):
    rand_indices = np.random.randint(0, num_cluster, size=batchsize)
    base_angle = math.pi * 2 / num_cluster
    angle = rand_indices * base_angle - math.pi / 2
    mean = np.zeros((batchsize, 2), dtype=np.float32)
    mean[:, 0] = np.cos(angle) * scale
    mean[:, 1] = np.sin(angle) * scale
    return np.random.normal(mean, std**2, (batchsize, 2)).astype(np.float32)

def build_discriminator(number=0):
    cnn = Sequential()
    cnn.add(Dense(256, input_shape=(2,)))
    cnn.add(LeakyReLU())
    cnn.add(Dense(128, input_shape=(2,)))
    cnn.add(LeakyReLU())
    cnn.add(Dense(128, input_shape=(2,)))
    cnn.add(LeakyReLU())

    point = Input(shape=(2,))
    features = cnn(point)
    is_real = Dense(1, activation='tanh',
                        name='generation_{0}'.format(number))(features)

    return Model(inputs=point, outputs=is_real)

def train(TRIAL_NUMBER):
    batchsize = 1024
    epoch_size = 500
    nb_epochs = 200
    nb_batches = nb_epochs * epoch_size

    discriminator = build_discriminator()
    discriminator.compile(optimizer=SGD(clipvalue=0.01),loss=[discriminator_loss])

    for batch in range(nb_batches):    
        fake_points = np.random.uniform(-4, 4, (batchsize, 2))
        real_points = gaussian_mixture_circle(batchsize)       

        X = np.concatenate((fake_points,real_points))
        y = np.array([1] * batchsize + [-0.5] * batchsize)

        disc_loss = discriminator.train_on_batch(X, y)

        if batch % epoch_size == epoch_size - 1:

            print("BATCH NUMBER " + str(batch))
            print("DISCRIMINATOR LOSS " + str(disc_loss))

            def plot_scatter(fake, true, dir=None, filename="scatter", color="blue"):
                fig = pylab.gcf()
                fig.set_size_inches(16.0, 16.0)
                pylab.clf()
                pylab.scatter(true[:, 0], true[:, 1], s=80, marker="X",
                              edgecolors="none", color='red')
                pylab.scatter(fake[:, 0], fake[:, 1], s=80, marker="o",
                              edgecolors="none", color='blue')

                h = .05  # step size in the mesh
                xx, yy = np.meshgrid(np.arange(-6, 6, h), np.arange(-6, 6, h))
                Z = discriminator.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                CS = pylab.contour(xx, yy, Z, 3, colors='k')
                pylab.clabel(CS, fontsize=9, inline=1)

                pylab.xlim(-5, 5)
                pylab.ylim(-5, 5)
                pylab.savefig("{}/{}/{}.png".format(
                    dir, TRIAL_NUMBER, filename))

            plot_scatter(fake_points, real_points, dir='plots/overfit',
                         filename='scatter{0:06d}'.format(batch))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        train(int(sys.argv[1]))
    else:
        train(0)

