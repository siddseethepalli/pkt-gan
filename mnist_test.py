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
NUM_CLUSTERS = 10
NUM_DISCRIMINATORS = 10


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
    cnn.add(Convolution2D(256, 5, 5, border_mode='same',
                          init='glorot_uniform'))
    cnn.add(LeakyReLU())

    # upsample to (..., 28, 28)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(128, 5, 5, border_mode='same',
                          init='glorot_uniform'))
    cnn.add(LeakyReLU())

    # take a channel axis reduction
    cnn.add(Convolution2D(1, 2, 2, border_mode='same',
                          activation='tanh', init='glorot_uniform'))

    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    image_class = Input(shape=(1,), dtype='int32')

    # 10 classes in MNIST
    cls = Flatten()(Embedding(10, latent_size,
                              init='glorot_uniform')(image_class))

    # hadamard product between z-space and a class conditional embedding
    h = merge([latent, cls], mode='mul')

    fake_image = cnn(h)

    return Model(input=[latent, image_class], output=fake_image)

def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    cnn = Sequential()
    #cnn.add(GaussianNoise(0.2, input_shape=(1, 28, 28)))
    cnn.add(Convolution2D(32, 3, 3, border_mode='same', subsample=(2, 2),
                          input_shape=(1, 28, 28)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(128, 3, 3, border_mode='same', subsample=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(256, 3, 3, border_mode='same', subsample=(1, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(1, 28, 28))

    features = cnn(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    predict = Dense(1, activation='sigmoid',
                    name='prediction_{0}'.format(number))(features)
    is_real = multiply([predict, Dense(1, activation='sigmoid',
                        name='generation_{0}'.format(number))(features)])
    aux_class = multiply([predict, Dense(NUM_CLUSTERS, activation='softmax',
                          name='auxiliary_{0}'.format(number))(features)])

    return Model(input=image, output=[predict, is_real, aux_class])


def gaussian_mixture_circle(batchsize, num_cluster=3, scale=3, std=0.5):
    rand_indices = np.random.randint(0, num_cluster, size=batchsize)
    base_angle = math.pi * 2 / num_cluster
    angle = rand_indices * base_angle - math.pi / 2
    mean = np.zeros((batchsize, 2), dtype=np.float32)
    mean[:, 0] = np.cos(angle) * scale
    mean[:, 1] = np.sin(angle) * scale
    return (np.random.normal(mean, std**2, (batchsize, 2)).astype(np.float32),
            rand_indices.astype(np.float32))

def auxiliary_loss(target, output):
    return K.mean(output * (1 - target))

def discriminator_loss(target, output):
    return K.mean(target*output)

def prediction_loss(target, output):
    return ALPHA * K.mean(target*output) * K.mean(target*output) / (K.mean(target*target) * K.mean(output*output) + 0.0001)

def combined_loss(target, output):
    output = K.clip(2 * output - 1, -1, 1)
    return K.mean(target, output)

def train(TRIAL_NUMBER):
    print('Training period started.')
    nb_discriminators = NUM_DISCRIMINATORS
    nb_epochs = 1000
    batch_size = 512
    epoch_size = 500
    nb_batches = nb_epochs * epoch_size
    latent_size = 100
    discriminators = []
    for num in range(nb_discriminators):
        discriminator = build_discriminator(num)
        discriminator.compile(
            optimizer=SGD(clipvalue=0.01),
            loss=[prediction_loss,
                  discriminator_loss,
                  auxiliary_loss]
        )
        discriminators.append(discriminator)

    generator = build_generator(latent_size)
    generator.compile(
        optimizer=SGD(clipvalue=0.01),
        loss='binary_crossentropy'
    )
    combineds = []
    for discriminator in discriminators:
        discriminator.trainable = False
        latent = Input(shape=(latent_size, ))
        cluster = Input(shape=(1,), dtype='int32')
        point = generator([latent, cluster])
        _, is_real, aux_class = discriminator(point)

        combined = Model(inputs=[latent, cluster], outputs=[is_real, aux_class])
        combined.compile(
            optimizer='RMSprop',
            loss=[discriminator_loss, auxiliary_loss]
        )
        combineds.append(combined)

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=1)

    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    X_test = np.expand_dims(X_test, axis=1)

    nb_train, nb_test = X_train.shape[0], X_test.shape[0]

    for batch in range(0, nb_batches):
        print('Batch {} of {}'.format(batch, nb_batches))

        point_batch = X_train[index * batch_size:(index + 1) * batch_size]
        cluster_batch = y_train[index * batch_size:(index + 1) * batch_size]

        noise = np.random.normal(0, 1, (batch_size, latent_size))
        sampled_clusters = np.random.randint(0, NUM_CLUSTERS, batch_size)
        generated_points = generator.predict(
            [noise, sampled_clusters.reshape((-1, 1))], verbose=0)

        X = np.concatenate((point_batch, generated_points))
        y_pred = np.ones(2 * batch_size)
        y = np.array([-1] * batch_size + [1] * batch_size)
        aux_y = np.concatenate((cluster_batch, sampled_clusters), axis=0).astype(np.int32)

        one_hots = np.zeros((2 * batch_size, NUM_CLUSTERS))
        for i in range(2*batch_size):
            one_hots[i][aux_y[i]] = 1

        disc_loss = []
        for i in range(nb_discriminators):
            for j in range(nb_discriminators):
                if i != j:
                    guess, _, _ = discriminators[j].predict_on_batch(X)
                    disc_loss.append(discriminators[i].train_on_batch(X, [guess, y, one_hots]))

        noise = np.random.normal(0, 1, (2 * batch_size, latent_size))
        sampled_clusters = np.random.randint(0, NUM_CLUSTERS, 2 * batch_size)

        trick = -np.ones(2 * batch_size)
        one_hots = np.zeros((2 * batch_size, NUM_CLUSTERS))
        for i in range(2*batch_size):
            one_hots[i][sampled_clusters[i]] = 1
        gen_loss = []
        for combined in combineds:
            gen_loss.append(combined.train_on_batch(
                [noise, sampled_clusters.reshape((-1, 1))],
                [trick, one_hots]
            ))

        print('full_disc_loss: {} full_gen_loss: {}'.format(disc_loss,
                                                            gen_loss))
        print('-------')
        print('sum_disc_loss: {} sum_gen_loss: {}'.format(sum([i[0] for i in disc_loss]), sum([i[0] for i in gen_loss])))

        for discriminator in discriminators:
            _, fake, _ = discriminator.predict_on_batch(X)
            if np.sum(fake) < 0.01:
                nb_discriminators -= 1
                discriminators.remove(discriminator)

        if batch % epoch_size == epoch_size - 1:
            generator.save_weights(
                'params/toy_params{1}_generator_batch_{0:06d}.hdf5'.format(
                    batch, TRIAL_NUMBER), True)
            discriminator.save_weights(
                'params/toy_params{1}_discriminator_batch_{0:06d}.hdf5'.format(
                    batch, TRIAL_NUMBER), True)

            noise = np.random.normal(-1, 1, (100, latent_size))

            sampled_labels = np.array([
                [i] * 10 for i in range(10)
            ]).reshape(-1, 1)

            # get a batch to display
            generated_images = generator.predict(
                [noise, sampled_labels], verbose=0)

            # arrange them into a grid
            img = (np.concatenate([r.reshape(-1, 28)
                                   for r in np.split(generated_images, 10)
                                   ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

            Image.fromarray(img).save(
                'plot_epoch_{0:03d}_generated.png'.format(epoch))



if __name__ == '__main__':
    if len(sys.argv) > 1:
        train(int(sys.argv[1]))
    else:
        train(0)
