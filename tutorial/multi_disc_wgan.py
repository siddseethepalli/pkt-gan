#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
from PIL import Image

from six.moves import range

import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout, average, concatenate, Lambda, add, dot, multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.noise import GaussianNoise
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.backend.common import _EPSILON
from keras.utils.generic_utils import Progbar
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
np.random.seed(1331)
K.set_image_dim_ordering('th')

ALPHA = 0
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
            repel_error += ALPHA / l2norm(s, s)
    return K.mean(target*output) + repel_error

def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 1, 28, 28)
    cnn = Sequential()

    cnn.add(Dense(1024, input_dim=latent_size))
    cnn.add(LeakyReLU())
    cnn.add(Dense(128 * 7 * 7))
    cnn.add(LeakyReLU())
    cnn.add(Reshape((128, 7, 7)))

    # upsample to (..., 14, 14)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Conv2D(256, (5, 5), padding='same', kernel_initializer='glorot_uniform'))
    cnn.add(LeakyReLU())

    # upsample to (..., 28, 28)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Conv2D(128, (5, 5), padding='same', kernel_initializer='glorot_uniform'))
    cnn.add(LeakyReLU())

    # take a channel axis reduction
    cnn.add(Conv2D(1, (2, 2), padding='same', activation='tanh', kernel_initializer='glorot_uniform'))

    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    image_class = Input(shape=(1,), dtype='int32')

    # 10 classes in MNIST
    cls = Flatten()(Embedding(10, 100, embeddings_initializer="glorot_uniform")(image_class))

    # hadamard product between z-space and a class conditional embedding
    h = multiply([latent, cls])

    fake_image = cnn(h)

    return Model(inputs=[latent, image_class], outputs=fake_image)

def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    cnn = Sequential()
    #cnn.add(GaussianNoise(0.2, input_shape=(1, 28, 28)))
    cnn.add(Conv2D(32, (3, 3), padding="same", strides=(2, 2), input_shape=(1, 28, 28)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(128, (3, 3), padding='same', strides=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(256, (3, 3), padding='same', strides=(1, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(1, 28, 28))

    features = cnn(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='linear', name='generation')(features)
    aux = Dense(10, activation='softmax', name='auxiliary')(features)

    return Model(inputs=image, outputs=[fake, aux])

def build_super_discriminator(discriminators):
    image = Input(shape=(1, 28, 28))

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

def train():
    # batch and latent size taken from the paper
    nb_batches = 100000
    batch_size = 128
    epoch_size = 20
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
        super_discriminator.load_weights('generator.hdf5')

    latent = Input(shape=(latent_size, ))
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

    # get our mnist data, and force it to be of shape (..., 1, 28, 28) with
    # range [-1, 1]
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=1)

    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    X_test = np.expand_dims(X_test, axis=1)

    nb_train, nb_test = X_train.shape[0], X_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    for batch in range(nb_batches):
        print('Batch {} of {}'.format(batch, nb_batches))

        idx = np.random.randint(X_train.shape[0] - batch_size)
        image_batch = X_train[idx: idx + batch_size]
        label_batch = y_train[idx: idx + batch_size]

        noise = np.random.normal(0, 1, (batch_size, latent_size))
        sampled_labels = np.random.randint(0, 10, batch_size)
        generated_images = generator.predict(
            [noise, sampled_labels.reshape((-1, 1))], verbose=0)

        X = np.concatenate((image_batch, generated_images))
        y = np.array([-1] * batch_size * NUM_DISCRIMINATORS + [1] * batch_size * NUM_DISCRIMINATORS).reshape((2 * batch_size, NUM_DISCRIMINATORS))
        aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

        disc_loss = super_discriminator.train_on_batch(X, [y, aux_y])

        noise = np.random.normal(0, 1, (2 * batch_size, latent_size))
        sampled_labels = np.random.randint(0, 10, 2 * batch_size)
        trick = -np.ones(2 * batch_size * NUM_DISCRIMINATORS).reshape((2 * batch_size, NUM_DISCRIMINATORS))

        gen_loss = combined.train_on_batch(
            [noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels])

        print('disc_loss: {} gen_loss: {}'.format(disc_loss, gen_loss))

        if batch % epoch_size == epoch_size - 1:
            generator.save_weights(
                'params_generator_batch_{0:06d}.hdf5'.format(batch), True)
            super_discriminator.save_weights(
                'params_discriminator_batch_{0:06d}.hdf5'.format(batch), True)

            noise = np.random.normal(-1, 1, (100, latent_size))
            sampled_labels = np.array([
                [i] * 10 for i in range(10)
            ]).reshape(-1, 1)

            generated_images = generator.predict(
                [noise, sampled_labels], verbose=0)

            img = (np.concatenate([r.reshape(-1, 28)
                                   for r in np.split(generated_images, 10)
                                   ], axis=-1) * 127.5 + 127.5).astype(np.uint8)
            Image.fromarray(img).save(
                'plot_batch_{0:06d}_generated.png'.format(batch))


if __name__ == '__main__':
    train()
