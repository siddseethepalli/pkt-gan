#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
from PIL import Image

import seaborn as sns
import pylab

from six.moves import range
import matplotlib
matplotlib.use('Agg')
import math 
import keras.backend as K
from keras.datasets import mnist
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.noise import GaussianNoise
from keras.losses import *
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.backend.common import _EPSILON
from keras.utils.generic_utils import Progbar
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plot

#plot.use('Agg')

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
K.set_image_dim_ordering('th')

TRIAL_NUMBER = int(sys.argv[1])

ALPHA = 0
LAMBDA = 0
NUM_DISCRIMINATORS = 3
NUM_CLUSTERS = 3
COLORS = ['r', 'green', 'blue']

def l2norm(a, b):
    return dot([a, b], 0)

def similar_disc_penalty(target, output):
    return ALPHA * (dot([target, output], axes=0) * dot([target, output], axes=0)) / (dot([target, target], axes=0) * dot([output, output], axes=0))

def modified_binary_crossentropy_disc(target, output):
    return K.mean(target*output)

def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 1, 28, 28)
    cnn = Sequential()

    cnn.add(Dense(20, input_shape=(latent_size,)))
    cnn.add(Activation(K.tanh))
    cnn.add(Dense(20))
    cnn.add(Activation(K.tanh))
    cnn.add(Dense(20))
    cnn.add(Activation(K.tanh))
    cnn.add(Dense(2, activation='linear'))


    # generator.add(BatchNormalization(128))
    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size,))

    # this will be our label
    image_class = Input(shape=(1,), dtype='int32')

    # 10 classes in MNIST
    cls = Flatten()(Embedding(NUM_CLUSTERS, latent_size, embeddings_initializer="glorot_uniform")(image_class))

    # hadamard product between z-space and a class conditional embedding
    h = multiply([latent, cls])

    fake_image = cnn(h)

    return Model(inputs=[latent, image_class], outputs=fake_image)

def build_discriminator(batch_size):
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    cnn = Sequential()
    cnn.add(Dense(64, input_shape=(2,)))
    cnn.add(LeakyReLU())
    cnn.add(Dense(64))
    cnn.add(LeakyReLU())
    cnn.add(Dense(32))
    cnn.add(LeakyReLU())

    image = Input(shape=(2,))

    features = cnn(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(NUM_CLUSTERS, activation='softmax', name='auxiliary')(features)

    return Model(inputs=image, outputs=[fake, aux, fake])

def build_super_discriminator(discriminators):
    image = Input(shape=(2,))

    fakes = []
    auxes = []
    for discriminator in discriminators:
        d_fake, d_aux, _ = discriminator(image)
        fakes.append(Reshape((-1,))(d_fake))
        auxes.append(d_aux)
    output = []
    normalizer = 0
    for fake in fakes:
        normalizer += K.sum(Lambda(lambda x: K.exp(tf.scalar_mul(LAMBDA,x)))(fake))
        temp = Lambda(lambda x: K.exp(tf.scalar_mul(LAMBDA,x)))(fake)
        fakes = multiply([temp, Reshape((3, -1, 1))(fake)])
    fakes = Lambda(lambda x: K.sum(x,axis=0)/normalizer)(fakes)
    print(fakes.shape)
    aux = average(auxes)

    return Model(inputs=image, outputs=[fakes, aux])

def gaussian_mixture_circle(batchsize, num_cluster=3, scale=3, std=0.5):
    rand_indices = np.random.randint(0, num_cluster, size=batchsize)
    base_angle = math.pi * 2 / num_cluster
    angle = rand_indices * base_angle - math.pi / 2
    mean = np.zeros((batchsize, 2), dtype=np.float32)
    mean[:, 0] = np.cos(angle) * scale
    mean[:, 1] = np.sin(angle) * scale
    return np.random.normal(mean, std**2, (batchsize, 2)).astype(np.float32), rand_indices.astype(np.float32)

def train(starting_batch):
    # batch and latent size taken from the paper
    nb_batches = 100000
    batch_size = 256
    epoch_size = 1000
    latent_size = 100

    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # build the super discriminator
    discriminators = []
    for _ in range(NUM_DISCRIMINATORS):
        discriminator = build_discriminator(batch_size)
        discriminator.compile(
            optimizer=SGD(clipvalue=0.01),#Adam(lr=adam_lr, beta_1=adam_beta_1),
            loss=[modified_binary_crossentropy_disc, 'sparse_categorical_crossentropy', similar_disc_penalty]
        )
        discriminators.append(discriminator)

    super_discriminator = build_super_discriminator(discriminators)
    super_discriminator.compile(optimizer=SGD(clipvalue=0.01),
            loss=[modified_binary_crossentropy_disc, 'sparse_categorical_crossentropy'])
    if os.path.exists('discriminator.hdf5'):
        super_discriminator.load_weights('discriminator.hdf5')

    # build the generator
    generator = build_generator(latent_size)
    generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
                      loss='binary_crossentropy')
    if os.path.exists('generator.hdf5'):
        generator.load_weights('generator.hdf5')

    latent = Input(shape=(latent_size,))
    image_class = Input(shape=(1,), dtype='int32')

    # get a fake image
    fake = generator([latent, image_class])

    # we only want to be able to train generation for the combined model
    super_discriminator.trainable = False
    fakes, aux = super_discriminator(fake)
    combined = Model(inputs=[latent, image_class], outputs=[fakes, aux])

    combined.compile(
        optimizer='RMSprop',
        loss=[modified_binary_crossentropy_disc,
              'sparse_categorical_crossentropy'])

    for batch in range(starting_batch, nb_batches):
        print('Batch {} of {}'.format(batch, nb_batches))

        image_batch, label_batch = gaussian_mixture_circle(batch_size)
        training_ys = np.zeros((NUM_DISCRIMINATORS, batch_size))
        for i in range(NUM_DISCRIMINATORS):
            #print([discriminators[i].predict(image_batch)[i].shape for i in range(3)])
            training_ys[i, :] = discriminators[i].predict(image_batch)[0].flatten()

        noise = np.random.normal(0, 1, (batch_size, latent_size))
        sampled_labels = np.random.randint(0, NUM_CLUSTERS, batch_size)
        generated_images = generator.predict(
            [noise, sampled_labels.reshape((-1, 1))], verbose=0)

        X = np.concatenate((image_batch, generated_images))
        y = np.array([-1] * batch_size + [1] * batch_size)
        aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

        def sdp(a,b):
            return abs(np.sum(a-b))


        #print(training_ys[0, :].shape)
        #print(y.shape)
        #print(sdp(training_ys[1,:],training_ys[2,:]))
        disc_loss = []
        order_1 = list(range(NUM_DISCRIMINATORS))
        np.random.shuffle(order_1)
        order_2 = list(range(NUM_DISCRIMINATORS))
        np.random.shuffle(order_2)
        for i in order_1:
            for j in order_2:
                if i != j:
                    disc_loss.append(discriminators[i].train_on_batch(X, [y, aux_y, y]))

        noise = np.random.normal(0, 1, (2 * batch_size, latent_size))
        sampled_labels = np.random.randint(0, NUM_CLUSTERS, 2 * batch_size)
        trick = -np.ones(2 * batch_size)

        gen_loss = combined.train_on_batch(
            [noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels])

        print('disc_loss: {} gen_loss: {}'.format(disc_loss, gen_loss))

        if batch % epoch_size == epoch_size - 1:
            generator.save_weights(
                'params/toy_params{1}_generator_batch_{0:06d}.hdf5'.format(batch,TRIAL_NUMBER), True)
            super_discriminator.save_weights(
                'params/toy_params{1}_discriminator_batch_{0:06d}.hdf5'.format(batch,TRIAL_NUMBER), True)
            
            def plot_scatter(fake, true, dir=None, filename="scatter",color="blue"):
                fig = pylab.gcf()
                fig.set_size_inches(16.0, 16.0)
                pylab.clf()
                pylab.scatter(true[:, 0], true[:, 1], s=80, marker="X", edgecolors="none", color='red')
                pylab.scatter(fake[:, 0], fake[:, 1], s=80, marker="o", edgecolors="none", color='blue')

                h = .02  # step size in the mesh
                xx, yy = np.meshgrid(np.arange(-4, 4, h),
                     np.arange(-4, 4, h))
                for i in range(NUM_DISCRIMINATORS):
                    Z, _, _ = discriminators[i].predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    CS = pylab.contour(xx, yy, Z, 3, colors=COLORS[i])
                    pylab.clabel(CS, fontsize=9, inline=1)

                Z, _ = super_discriminator.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                CS = pylab.contour(xx, yy, Z, 3, colors='k')
                pylab.clabel(CS, fontsize=9, inline=1)

                pylab.xlim(-4, 4)
                pylab.ylim(-4, 4)
                pylab.savefig("{}/{}.png".format(dir, filename))

            plot_scatter(generated_images,image_batch,dir='plots',filename='scatter{0:06d}'.format(batch))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        train(int(sys.argv[1]))
    else:
        train(0)
