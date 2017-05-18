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

COLORS = ['r','g','b','c','m','y','k']
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
K.set_image_dim_ordering('th')

NUM_CLUSTERS = 3
ALPHA = 2
NUM_DISCRIMINATORS = 3

#GENERATOR
def build_generator(latent_size):
	#Builds the model
	cnn = Sequential()
	cnn.add(Dense(50, input_shape=(latent_size,)))
	cnn.add(Activation(K.tanh))
	cnn.add(Dense(25))
	cnn.add(Activation(K.tanh))
	cnn.add(Dense(10))
	cnn.add(Activation(K.tanh))
	cnn.add(Dense(2, activation='linear'))

	#Input of random noise (z-space)
	latent = Input(shape=(latent_size, ))

	#Defines the image class from which we will select
	cluster = Input(shape=(1,), dtype='int32')

	#Embeds the classes in the latent space
	cls = Flatten()(Embedding(NUM_CLUSTERS, 100, embeddings_initializer="glorot_uniform")(cluster))
	h = multiply([latent, cls])

	#Creates a point out of the noise
	generated_point = cnn(h)

	return Model(inputs=[latent, cluster], outputs=generated_point)

#DISCRIMINATOR
def build_discriminator(number):
	#Builds the model
	cnn = Sequential()
	cnn.add(Dense(128, input_shape=(2,)))
	cnn.add(Activation('relu'))
	cnn.add(Dense(64))
	cnn.add(LeakyReLU())
	
	#Input point (x,y)
	point = Input(shape=(2,))

	features = cnn(point)

	#Outputs is_fake, aux_class
	is_fake = Dense(1, activation='sigmoid', name='generation_{0}'.format(number))(features)
	aux_class = Dense(NUM_CLUSTERS, activation='softmax', name='auxiliary_{0}'.format(number))(features)

	return Model(inputs=point, outputs=[is_fake, aux_class, is_fake])

#SUPERDISCRIMINATOR
def build_superdiscriminator(discriminators):
	point = Input(shape=(2,))
	fakes = []
	auxes = []
	for discriminator in discriminators:
		d_fake, d_aux, _ = discriminator(point)
		fakes.append(Reshape((-1,))(d_fake))
		auxes.append(d_aux)
	is_fake = maximum(fakes)
	aux_class = average(auxes)

	return Model(inputs=point, outputs=[is_fake, aux_class])

#REAL DISTRIBUTION
def gaussian_mixture_circle(batchsize, num_cluster=3, scale=3, std=0.5):
	rand_indices = np.random.randint(0, num_cluster, size=batchsize)
	base_angle = math.pi * 2 / num_cluster
	angle = rand_indices * base_angle - math.pi / 2
	mean = np.zeros((batchsize, 2), dtype=np.float32)
	mean[:, 0] = np.cos(angle) * scale
	mean[:, 1] = np.sin(angle) * scale
	return np.random.normal(mean, std**2, (batchsize, 2)).astype(np.float32), rand_indices.astype(np.float32)

#LOSSES
def discriminator_loss(target, output):
	return -K.mean(target*output)/NUM_DISCRIMINATORS

def discriminator_repulsion(target, output):
	return ALPHA*K.mean(target*output)*K.mean(target*output)/(K.mean(target*target)*K.mean(output*output))

#TRAINING
def train(TRIAL_NUMBER):
	print('Training period started.')
	#batch and latent size taken from the paper
	nb_epochs = 10
	batch_size = 512
	epoch_size = 200
	nb_batches = nb_epochs * epoch_size
	latent_size = 100

	#Adam parameters suggested in https://arxiv.org/abs/1511.06434
	adam_lr = 0.0002
	adam_beta_1 = 0.5

	#build the discriminators
	discriminators = []
	for number in range(NUM_DISCRIMINATORS):
		discriminator = build_discriminator(number)
		discriminator.compile(
			optimizer=SGD(clipvalue=0.01),
			loss=[discriminator_loss, 'sparse_categorical_crossentropy', discriminator_repulsion])
		discriminators.append(discriminator)

	#build the generator
	generator = build_generator(latent_size)
	generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
					  loss='binary_crossentropy')

	#build superdiscriminator
	superdiscriminator = build_superdiscriminator(discriminators)
	superdiscriminator.compile(optimizer=SGD(clipvalue=0.01),
		   loss=[discriminator_loss, 'sparse_categorical_crossentropy'])

	#get inputs, etc.
	latent = Input(shape=(latent_size, ))
	cluster = Input(shape=(1,), dtype='int32')
	point = generator([latent, cluster])

	superdiscriminator.trainable = False
	is_fake, aux_class = superdiscriminator(point)

	#build simple combined model
	combined = Model(inputs=[latent, cluster], outputs=[is_fake, aux_class])
	combined.compile(
		optimizer='RMSprop',
		loss=[discriminator_loss, 'sparse_categorical_crossentropy'])

	for batch in range(0, nb_batches):
		print('Batch {} of {}'.format(batch, nb_batches))

		#idx = np.random.randint(X_train.shape[0] - batch_size)
		point_batch, cluster_batch = gaussian_mixture_circle(batch_size,num_cluster=NUM_CLUSTERS)
		noise = np.random.normal(0, 1, (batch_size, latent_size))
		sampled_clusters = np.random.randint(0, NUM_CLUSTERS, batch_size)
		generated_points = generator.predict([noise, sampled_clusters.reshape((-1, 1))], verbose=0)
		
		KAPPA = 3 * (nb_batches - batch) / nb_batches + 1

		X = np.concatenate((point_batch, generated_points))
		y = np.array([1] * batch_size * NUM_DISCRIMINATORS + [-KAPPA] * batch_size * NUM_DISCRIMINATORS).reshape((2 * batch_size, NUM_DISCRIMINATORS))
		aux_y = np.concatenate((cluster_batch, sampled_clusters), axis=0)

		disc_loss = []
		for i in range(NUM_DISCRIMINATORS):
			for j in range(NUM_DISCRIMINATORS):
				if i != j:
					guess, _, _ = discriminators[j].predict_on_batch(X)
					disc_loss.append(discriminators[i].train_on_batch(X, [y, aux_y, guess]))

		noise = np.random.normal(0, 1, (2 * batch_size, latent_size))
		sampled_clusters = np.random.randint(0, NUM_CLUSTERS, 2 * batch_size)
		trick = np.ones(2 * batch_size * NUM_DISCRIMINATORS).reshape((2 * batch_size, NUM_DISCRIMINATORS))

		gen_loss = combined.train_on_batch([noise, sampled_clusters.reshape((-1, 1))], [trick, sampled_clusters])

		print('full_disc_loss: {} full_gen_loss: {}'.format(disc_loss, gen_loss))
		print('disc_loss: {} gen_loss: {}'.format(sum([i[0] for i in disc_loss]), gen_loss[0]))

		if batch % epoch_size == epoch_size - 1:
			generator.save_weights(
				'params/toy_params{1}_generator_batch_{0:06d}.hdf5'.format(batch,TRIAL_NUMBER), True)
			for i in range(NUM_DISCRIMINATORS):
				discriminator.save_weights(
					'params/toy_params{1}_discriminator_{2}_batch_{0:06d}.hdf5'.format(batch,TRIAL_NUMBER,i), True)
			
			def plot_scatter(fake, true, dir=None, filename="scatter",color="blue"):
				fig = pylab.gcf()
				fig.set_size_inches(16.0, 16.0)
				pylab.clf()
				pylab.scatter(true[:, 0], true[:, 1], s=80, marker="X", edgecolors="none", color='red')
				pylab.scatter(fake[:, 0], fake[:, 1], s=80, marker="o", edgecolors="none", color='blue')

				h = .02  # step size in the mesh
				xx, yy = np.meshgrid(np.arange(-6, 6, h),
					 np.arange(-6, 6, h))
				for i in range(NUM_DISCRIMINATORS):
					Z, _, _ = discriminators[i].predict(np.c_[xx.ravel(), yy.ravel()])
					Z = Z.reshape(xx.shape)
					CS = pylab.contour(xx, yy, Z, 3, colors=COLORS[i])
					pylab.clabel(CS, fontsize=9, inline=1)

				Z, _ = superdiscriminator.predict(np.c_[xx.ravel(), yy.ravel()])
				Z = Z.reshape(xx.shape)
				CS = pylab.contour(xx, yy, Z, 3, colors='k')
			   #pylab.clabel(CS, fontsize=9, inline=1)

				pylab.xlim(-6, 6)
				pylab.ylim(-6, 6)
				pylab.savefig("{}/{}/{}.png".format(dir, TRIAL_NUMBER, filename))

			plot_scatter(generated_points,point_batch,dir='plots',filename='scatter{0:06d}'.format(batch))
	
if __name__ == '__main__':
	if len(sys.argv) > 1:
		train(int(sys.argv[1]))
	else:
		train(0)
