#!/usr/bin/env python
import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, average, multiply, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.core import Lambda
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.backend.common import _EPSILON
from keras.utils.generic_utils import Progbar
import numpy as np
from PIL import Image

np.random.seed(1331)
K.set_image_dim_ordering('th')

def modified_binary_crossentropy(target, output):
    return K.mean(target*output)

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
    cnn.add(Conv2D(1, (2, 2), padding='same', activation='tanh',
                   kernel_initializer='glorot_uniform'))

    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    image_class = Input(shape=(1,), dtype='int32')

    # 10 classes in MNIST
    cls = Flatten()(Embedding(10, latent_size,
                              embeddings_initializer='glorot_uniform')(image_class))

    # hadamard product between z-space and a class conditional embedding
    h = multiply([latent, cls])

    fake_image = cnn(h)

    return Model(outputs=[fake_image], inputs=[latent, image_class])

def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    cnn = Sequential()
    #cnn.add(GaussianNoise(0.2, input_shape=(1, 28, 28)))
    cnn.add(Conv2D(32, (3, 3), padding='same', strides=(2, 2),
                          input_shape=(1, 28, 28)))
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

    return Model(outputs=[fake, aux], inputs=image)

def train():
    # batch and latent size taken from the paper
    nb_epochs = 50
    batch_size = 10
    latent_size = 100

    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # build the first discriminator
    first_discriminator = build_discriminator()
    first_discriminator.compile(
        optimizer=SGD(clipvalue=0.01),#Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=[modified_binary_crossentropy, 'sparse_categorical_crossentropy']
    )

    # build the second discriminator
    second_discriminator = build_discriminator()
    second_discriminator.compile(
        optimizer=SGD(clipvalue=0.01),#Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=[modified_binary_crossentropy, 'sparse_categorical_crossentropy']
    )

    # build the generator
    generator = build_generator(latent_size)
    generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
                      loss='binary_crossentropy')

    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1,), dtype='int32')

    # get a fake image
    fake = generator([latent, image_class])

    first_discriminator.trainable = False
    second_discriminator.trainable = False
    fake_1, aux_1 = first_discriminator(fake)
    fake_2, aux_2 = second_discriminator(fake)
    aux = average([aux_1, aux_2])
    fake = multiply([Lambda(lambda x: 0.5 * (x + 1))(fake_1),
                     Lambda(lambda x: 0.5 * (x + 1))(fake_2)])
    combined = Model(outputs=[fake, aux], inputs=[latent, image_class])

    combined.compile(
        optimizer='RMSprop',
        loss=[modified_binary_crossentropy, 'sparse_categorical_crossentropy']
    )

    # get our mnist data, and force it to be of shape (..., 1, 28, 28) with
    # range [-1, 1]
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=1)

    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    X_test = np.expand_dims(X_test, axis=1)

    nb_train, nb_test = X_train.shape[0], X_test.shape[0]

    for epoch in range(nb_epochs):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = int(X_train.shape[0] / batch_size)
        progress_bar = Progbar(target=nb_batches)

        epoch_gen_loss = []
        epoch_disc_1_loss = []
        epoch_disc_2_loss = []

        for index in range(nb_batches):
            if len(epoch_gen_loss) + len(epoch_disc_1_loss) + len(epoch_disc_2_loss) >= 1:
                progress_bar.update(
                    index, values=[
                        ('disc_1_loss',np.mean(np.array(epoch_disc_1_loss),axis=0)[0]),
                        ('disc_2_loss',np.mean(np.array(epoch_disc_2_loss),axis=0)[0]),
                        ('gen_loss', np.mean(np.array(epoch_gen_loss),axis=0)[0])
                    ])
            else:
                progress_bar.update(index)

            noise = np.random.normal(0, 1, (batch_size, latent_size))
            sampled_labels = np.random.randint(0, 10, batch_size)

            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]

            generated_images = generator.predict(
                [noise, sampled_labels.reshape((-1, 1))], verbose=0)

            X = np.concatenate((image_batch, generated_images))
            y = np.array([-1] * batch_size + [1] * batch_size)
            aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

            epoch_disc_1_loss.append(first_discriminator.train_on_batch(X, [y, aux_y]))

            fake_predicted_a,_ = first_discriminator.predict_on_batch(image_batch)
            fake_predicted_b,_ = first_discriminator.predict_on_batch(generated_images)
            
            fake_predicted_a,fake_predicted_b = fake_predicted_a.flatten(),fake_predicted_b.flatten()

            mask_a, mask_b = fake_predicted_a>0, 1*fake_predicted_b<0
            X_wrong = np.concatenate((image_batch[mask_a],generated_images[mask_b]))
            y_wrong =np.concatenate((np.ones(image_batch[mask_a].shape[0]),-1*np.ones(generated_images[mask_b].shape[0])))
            aux_y_wrong = np.concatenate((label_batch[mask_a],sampled_labels[mask_b]))

            epoch_disc_2_loss.append(
                second_discriminator.train_on_batch(X_wrong, [y_wrong, aux_y_wrong]))

            noise = np.random.normal(0, 1, (2 * batch_size, latent_size))
            sampled_labels = np.random.randint(0, 10, 2 * batch_size)
            trick = -np.ones(2 * batch_size)

            epoch_gen_loss.append(combined.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels]))

        generator.save_weights(
            'params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
        first_discriminator.save_weights(
            'params_discriminator_1_epoch_{0:03d}.hdf5'.format(epoch), True)
        second_discriminator.save_weights(
            'params_discriminator_2_epoch_{0:03d}.hdf5'.format(epoch), True)

        noise = np.random.normal(-1, 1, (100, latent_size))
        sampled_labels = np.array([
            [i] * 10 for i in range(10)
        ]).reshape(-1, 1)
        generated_images = generator.predict([noise, sampled_labels], verbose=0)
        img = (np.concatenate([r.reshape(-1, 28)
                               for r in np.split(generated_images, 10)
                               ], axis=-1) * 127.5 + 127.5).astype(np.uint8)
        Image.fromarray(img).save(
            'plot_epoch_{0:03d}_generated.png'.format(epoch))

if __name__ == '__main__':
    train()
