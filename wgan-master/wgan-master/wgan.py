import os
import time
import argparse
import importlib
import tensorflow as tf
import tensorflow.contrib as tc

from visualize import *


class WassersteinGAN(object):
    def __init__(self, g_net, d1_net, d2_net, d3_net, x_sampler, z_sampler, data, model):
        self.model = model
        self.data = data
        self.g_net = g_net
        self.d1_net = d1_net
        self.d2_net = d2_net
        self.d3_net = d3_net
        self.x_sampler = x_sampler
        self.z_sampler = z_sampler
        self.x_dim = self.d1_net.x_dim
        self.z_dim = self.g_net.z_dim
        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.z_ = self.g_net(self.z)

        self.d1 = self.d1_net(self.x, reuse=False)
        self.d2 = self.d2_net(self.x, reuse=False)
        self.d3 = self.d3_net(self.x, reuse=False)

        self.d1_ = self.d1_net(self.z_)
        self.d2_ = self.d2_net(self.z_)
        self.d3_ = self.d3_net(self.z_)

        self.g_loss = tf.reduce_mean(tf.add_n([self.d1_,self.d2_,self.d3_]))
        self.d1_loss = tf.reduce_mean(self.d1) - tf.reduce_mean(self.d1_)
        self.d2_loss = tf.reduce_mean(self.d2) - tf.reduce_mean(self.d2_)
        self.d3_loss = tf.reduce_mean(self.d3) - tf.reduce_mean(self.d3_)

        self.reg = tc.layers.apply_regularization(
            tc.layers.l1_regularizer(2.5e-5),
            weights_list=[var for var in tf.global_variables() if 'weights' in var.name]
        )
        self.g_loss_reg = self.g_loss + self.reg
        self.d1_loss_reg = self.d1_loss + self.reg
        self.d2_loss_reg = self.d2_loss + self.reg
        self.d3_loss_reg = self.d3_loss + self.reg

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d1_rmsprop = tf.train.RMSPropOptimizer(learning_rate=5e-5)\
                .minimize(self.d1_loss_reg, var_list=self.d1_net.vars)
            self.d2_rmsprop = tf.train.RMSPropOptimizer(learning_rate=5e-5)\
                .minimize(self.d2_loss_reg, var_list=self.d2_net.vars)
            self.d3_rmsprop = tf.train.RMSPropOptimizer(learning_rate=5e-5)\
                .minimize(self.d3_loss_reg, var_list=self.d3_net.vars)
            self.g_rmsprop = tf.train.RMSPropOptimizer(learning_rate=5e-5)\
                .minimize(self.g_loss_reg, var_list=self.g_net.vars)

        self.d1_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.d1_net.vars]
        self.d2_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.d2_net.vars]
        self.d3_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.d3_net.vars]
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, batch_size=64, num_batches=1000000):
        plt.ion()
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for t in range(0, num_batches):
            d_iters = 5
            if t % 500 == 0 or t < 25:
                 d_iters = 100

            for _ in range(0, d_iters):
                bx = self.x_sampler(batch_size)
                bz = self.z_sampler(batch_size, self.z_dim)
                self.sess.run(self.d1_clip)
                self.sess.run(self.d2_clip)
                self.sess.run(self.d3_clip)

                self.sess.run(self.d1_rmsprop, feed_dict={self.x: bx, self.z: bz})
                self.sess.run(self.d2_rmsprop, feed_dict={self.x: bx, self.z: bz})
                self.sess.run(self.d3_rmsprop, feed_dict={self.x: bx, self.z: bz})

            bz = self.z_sampler(batch_size, self.z_dim)
            self.sess.run(self.g_rmsprop, feed_dict={self.z: bz, self.x: bx})

            if t % 100 == 0:
                bx = self.x_sampler(batch_size)
                bz = self.z_sampler(batch_size, self.z_dim)

                d1_loss = self.sess.run(
                    self.d1_loss, feed_dict={self.x: bx, self.z: bz}
                )
                d2_loss = self.sess.run(
                    self.d2_loss, feed_dict={self.x: bx, self.z: bz}
                )
                d3_loss = self.sess.run(
                    self.d3_loss, feed_dict={self.x: bx, self.z: bz}
                )
                g_loss = self.sess.run(
                    self.g_loss, feed_dict={self.z: bz, self.x: bx}
                )
                print('Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' %
                        (t, time.time() - start_time, d_loss - g_loss, g_loss))

            if t % 100 == 0:
                bz = self.z_sampler(batch_size, self.z_dim)
                bx = self.sess.run(self.x_, feed_dict={self.z: bz})
                bx = xs.data2img(bx)
                fig = plt.figure(self.data + '.' + self.model)
                grid_show(fig, bx, xs.shape)
                fig.savefig('logs/{}/{}.pdf'.format(self.data, t/100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--model', type=str, default='dcgan')
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    data = importlib.import_module(args.data)
    model = importlib.import_module(args.data + '.' + args.model)
    xs = data.DataSampler()
    zs = data.NoiseSampler()
    d1_net = model.Discriminator(1)
    d2_net = model.Discriminator(2)
    d3_net = model.Discriminator(3)
    g_net = model.Generator()
    wgan = WassersteinGAN(g_net, d1_net, d2_net, d3_net, xs, zs, args.data, args.model)
    wgan.train()
