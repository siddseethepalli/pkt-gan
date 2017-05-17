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
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.backend.common import _EPSILON
from keras.utils.generic_utils import Progbar
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plot

#plot.use('Agg')

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
np.random.seed(1331)
K.set_image_dim_ordering('th')

TRIAL_NUMBER = int(sys.argv[1])

ALPHA = 1e-3
NUM_DISCRIMINATORS = 3

def l2norm(a, b):
    return dot([a, b], 0)

def modified_binary_crossentropy_disc(target, output):
    return K.mean(target*output)

def modified_binary_crossentropy_super(target, output):
    repel_error = K.constant(0)
    for i in range(NUM_DISCRIMINATORS):
        for j in range(i + 1, NUM_DISCRIMINATORS):
            s = Reshape((-1, 1))(output[:, i] - output[:, j])
            #repel_error += ALPHA / l2norm(s, s)
            #a, b = Reshape((-1, 1))(target[:, i]), Reshape((-1, 1))(output[:, j])
            #repel_error += ALPHA * (l2norm(a, b)/(l2norm(a, a)*l2norm(b, b)))
    return K.mean(target*output) #+ repel_error

def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 1, 28, 28)
    cnn = Sequential()

    cnn.add(Activation('relu',input_shape=(latent_size,)))
    cnn.add(Activation('relu'))
    cnn.add(Dense(2,activation='linear'))


    # generator.add(BatchNormalization(128))
    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    image_class = Input(shape=(1,), dtype='int32')

    # 10 classes in MNIST
    cls = Flatten()(Embedding(4, 100, embeddings_initializer="glorot_uniform")(image_class))

    # hadamard product between z-space and a class conditional embedding
    h = multiply([latent, cls])

    fake_image = cnn(h)

    return Model(inputs=[latent, image_class], outputs=fake_image)

def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    cnn = Sequential()
    #cnn.add(GaussianNoise(0.2, input_shape=(1, 28, 28)))
    cnn.add(Dense(8, activation='linear', input_shape=(1,2)))
    cnn.add(Activation(K.tanh))

    image = Input(shape=(2,))

    features = cnn(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='linear', name='generation')(features)
    aux = Dense(4, activation='softmax', name='auxiliary')(features)
    print(fake.shape, aux.shape)
    return Model(inputs=image, outputs=[fake, aux])

def build_super_discriminator(discriminators):
    image = Input(shape=(2,))

    fakes = []
    auxes = []
    for discriminator in discriminators:
        d_fake, d_aux = discriminator(image)
        fakes.append(Reshape((-1,))(d_fake))
        auxes.append(d_aux)

    fake = concatenate(fakes)
    # out = []
    # for i in range(256):
    #     out.append(tf.strided_slice(fake,i,2*256+i,256))
    # fake = tf.stack(out)
    aux = average(auxes)
    
    # def l2norm(a, b):
    #     return Reshape(())(dot([a - b, a - b], 0))

    # diffs = []
    # for i in range(len(discriminators)):
    #     for j in range(i + 1, len(discriminators)):
    #         diffs.append(l2norm(fakes[i], fakes[j]))

    # diff = concatenate(diffs)

    return Model(inputs=image, outputs=[fake, aux])

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
    batch_size = 128
    epoch_size = 100
    latent_size = 100 

    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # build the super discriminator
    discriminators = []
    for _ in range(NUM_DISCRIMINATORS):
        discriminator = build_discriminator()
        discriminator.compile(
            optimizer=SGD(clipvalue=0.01),#Adam(lr=adam_lr, beta_1=adam_beta_1),
            loss=[modified_binary_crossentropy_disc, 'sparse_categorical_crossentropy']
        )
        discriminators.append(discriminator)

    super_discriminator = build_super_discriminator(discriminators)
    super_discriminator.compile(optimizer=SGD(clipvalue=0.01),
            loss=[modified_binary_crossentropy_super, 'sparse_categorical_crossentropy'])
    if os.path.exists('discriminator.hdf5'):
        super_discriminator.load_weights('discriminator.hdf5')

    # build the generator
    generator = build_generator(latent_size)
    generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
                      loss='binary_crossentropy')
    if os.path.exists('generator.hdf5'):
        generator.load_weights('generator.hdf5')

    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1,), dtype='int32')

    # get a fake image
    fake = generator([latent, image_class])
    print(fake.shape)

    # we only want to be able to train generation for the combined model
    super_discriminator.trainable = False
    fakes, aux = super_discriminator(fake)
    combined = Model(inputs=[latent, image_class], outputs=[fakes, aux])

    combined.compile(
        optimizer='RMSprop',
        loss=[modified_binary_crossentropy_disc,
              'sparse_categorical_crossentropy'])

    # get our mnist data, and force it to be of shape (..., 1, 28, 28) with
    # range [-1, 1]
    #(X_train, y_train), (X_test, y_test) = sample()
    #X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    #X_train = np.expand_dims(X_train, axis=1)

    #X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    #X_test = np.expand_dims(X_test, axis=1)

    #nb_train, nb_test = X_train.shape[0], X_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    for batch in range(starting_batch, nb_batches):
        print('Batch {} of {}'.format(batch, nb_batches))

        #idx = np.random.randint(X_train.shape[0] - batch_size)
        image_batch, label_batch = gaussian_mixture_circle(batch_size)
        noise = np.random.normal(0, 1, (batch_size, latent_size))
        sampled_labels = np.random.randint(0, 3, batch_size)
        generated_images = generator.predict(
            [noise, sampled_labels.reshape((-1, 1))], verbose=0)

        X = np.concatenate((image_batch, generated_images))
        y = np.array([-1] * batch_size * NUM_DISCRIMINATORS + [1] * batch_size * NUM_DISCRIMINATORS).reshape((2 * batch_size, NUM_DISCRIMINATORS))
        aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

        disc_loss = super_discriminator.train_on_batch(X, [y, aux_y])

        noise = np.random.normal(0, 1, (2 * batch_size, latent_size))
        sampled_labels = np.random.randint(0, 3, 2 * batch_size)
        trick = -np.ones(2 * batch_size * NUM_DISCRIMINATORS).reshape((2 * batch_size, NUM_DISCRIMINATORS))

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
                pylab.scatter(fake[:, 0], fake[:, 1], s=80, marker="o", edgecolors="none", color='blue')
                pylab.scatter(true[:, 0], true[:, 1], s=80, marker="X", edgecolors="none", color='red')
                pylab.xlim(-4, 4)
                pylab.ylim(-4, 4)
                pylab.savefig("{}/{}.png".format(dir, filename))

            plot_scatter(generated_images,image_batch,dir='plots',filename='scatter{0}'.format(batch))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        train(int(sys.argv[1]))
    else:
        train(0)
