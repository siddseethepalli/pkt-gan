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
NUM_DISCRIMINATORS = 3


def build_generator(latent_size):
    cnn = Sequential()
    cnn.add(Dense(20, input_shape=(latent_size,)))
    cnn.add(Activation(K.tanh))
    cnn.add(Dense(20))
    cnn.add(Activation(K.tanh))
    cnn.add(Dense(20))
    cnn.add(Activation(K.tanh))
    cnn.add(Dense(2, activation='linear'))

    latent = Input(shape=(latent_size,))
    cluster = Input(shape=(1,), dtype='int32')
    cls = Flatten()(Embedding(NUM_CLUSTERS, latent_size,
                    embeddings_initializer="glorot_uniform")(cluster))
    h = multiply([latent, cls])
    generated_point = cnn(h)

    return Model(inputs=[latent, cluster], outputs=generated_point)


def build_discriminator(number=0):
    cnn = Sequential()
    cnn.add(Dense(256, input_shape=(2,)))
    cnn.add(LeakyReLU())
    cnn.add(Dense(128))
    cnn.add(LeakyReLU())

    point = Input(shape=(2,))
    features = cnn(point)
    predict = Dense(1, activation='sigmoid',
                    name='prediction_{0}'.format(number))(features)
    is_real = multiply([predict, Dense(1, activation='relu',
                        name='generation_{0}'.format(number))(features)])
    aux_class = multiply([predict, Dense(NUM_CLUSTERS, activation='softmax',
                          name='auxiliary_{0}'.format(number))(features)])

    return Model(inputs=point, outputs=[predict, is_real, aux_class])


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

    for batch in range(0, nb_batches):
        print('Batch {} of {}'.format(batch, nb_batches))

        point_batch, cluster_batch = gaussian_mixture_circle(
            batch_size, num_cluster=NUM_CLUSTERS)
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

        for i in range(nb_discriminators):
            _, fake, _ = discriminators[i].predict_on_batch(X)
            if np.sum(fake) < 0.05:
                nb_discriminators -= 1
                discriminators.remove(i)

        if batch % epoch_size == epoch_size - 1:
            generator.save_weights(
                'params/toy_params{1}_generator_batch_{0:06d}.hdf5'.format(
                    batch, TRIAL_NUMBER), True)
            discriminator.save_weights(
                'params/toy_params{1}_discriminator_batch_{0:06d}.hdf5'.format(
                    batch, TRIAL_NUMBER), True)

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
                for i in range(nb_discriminators):
                    Z, _, _ = discriminators[i].predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    CS = pylab.contour(xx, yy, Z, 3, colors=COLORS[i])
                    pylab.clabel(CS, fontsize=9, inline=1)

                pylab.xlim(-6, 6)
                pylab.ylim(-6, 6)
                pylab.savefig("{}/{}/{}.png".format(
                    dir, TRIAL_NUMBER, filename))

            plot_scatter(generated_points, point_batch, dir='plots',
                         filename='scatter{0:06d}'.format(batch))



if __name__ == '__main__':
    if len(sys.argv) > 1:
        train(int(sys.argv[1]))
    else:
        train(0)
