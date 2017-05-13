#!/usr/bin/env python
import tensorflow as tf

def tutorial():
    node1 = tf.constant(3.0)
    node2 = tf.constant(4.0)
    print(node1, node2)

    sess = tf.Session()
    print(sess.run([node1, node2]))

    node3 = tf.add(node1, node2)


if __name__ == '__main__':
    tutorial()
