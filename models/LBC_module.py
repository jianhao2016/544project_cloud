#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This is the local binary pattern module.
    -- input : x - a tensor from previous layer
    -- output : y - a tensor, similar to what convolution 
                    layer outputs
"""
# import tensorflow as tf
# import numpy as np
# 
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

FLAGS = None

def LBC(x, number_of_b, sparsity,
        filter_height, filter_width,
        input_channels, output_channels,
        data_format, strides = 1, padding = 'SAME',
        shared_weights = False):
    """
    performs the LBC function,
    input   -> x : batch of images
    output  -> y_LBC: batch of convoluted images
    """
    # setup local variables, i.e. shape, length, etc.
    ancher_shape = np.array([filter_height, filter_width,
                            input_channels, output_channels])
    num_elements = np.product(ancher_shape, dtype = int)
    num_non_zero = num_elements * sparsity
    num_non_zero = num_non_zero.astype(int)

    # initialize ancher weights
    ancher_weights = np.zeros(shape = ancher_shape, dtype = np.float)
    ancher_weights = np.reshape(ancher_weights, newshape = [num_elements])
    if shared_weights:
        np.random.seed(seed = 42)
    index_non_zero = np.random.choice(num_elements, num_non_zero, replace = False)
    for i in index_non_zero:
        ancher_weights[i] = np.random.binomial(1, 0.5) * 2 - 1
    ancher_weights = np.reshape(ancher_weights, newshape = ancher_shape)
    ancher_weights_tensor = tf.constant(ancher_weights, dtype = tf.float32)

    # if data_format == 'NCHW':
    #     strides = strides
    # else:
    #     strides = [1, strdes, strides, 1]
    # diffmap = tf.layers.conv2d(x, filters = output_channels, 
    #                            kernel_size = filter_height,
    #                            strides = strides,
    #                            padding = padding,
    #                            data_format = data_format,
    #                            use_bias = False)
    if data_format == 'channels_first':
        tf_format = 'NCHW'
        tf_strides = [1, 1, strides, strides]
    else:
        tf_format = 'NHWC'
        tf_strides = [1, strides, strides, 1]
    diffmap = tf.nn.conv2d(input = x, filter = ancher_weights_tensor, 
                          strides = tf_strides,
                          padding = padding, data_format = tf_format)
    bitmap = tf.sigmoid(diffmap)

    # print(x.get_shape().as_list())
    # print(ancher_weights_tensor.get_shape().as_list())
    y_LBC = tf.layers.conv2d(inputs = bitmap,
            filters = output_channels,
            kernel_size = [1,1],
            padding = padding,
            data_format = data_format,
            use_bias = False)
    return y_LBC

def deepnn(x):
    x_reshape = tf.reshape(x, [-1, 28, 28, 1])
    # h_conv1 = tf.contrib.layers.conv2d(x_reshape, 
    #         num_outputs = 32, 
    #         kernel_size = [5, 5],
    #         padding = 'SAME')
    h_conv1 = LBC(x_reshape, 512, 0.5, 5, 5, 1, 32)
    h_pool1 = tf.nn.avg_pool(h_conv1, ksize = [1, 2, 2, 1],
                            strides = [1, 2, 2, 1],
                            padding = 'SAME')
    
    h_conv2 = tf.contrib.layers.conv2d(h_pool1, 
            num_outputs = 64, 
            kernel_size = [5, 5],
            padding = 'SAME')
    h_pool2 = tf.nn.avg_pool(h_conv2, ksize = [1, 2, 2, 1],
                            strides = [1, 2, 2, 1],
                            padding = 'SAME')

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.contrib.layers.fully_connected(h_pool2_flat, 
                                             num_outputs = 1024)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    y_cnn = tf.contrib.layers.fully_connected(h_fc1_drop,
                                            num_outputs = 10)

    return y_cnn, keep_prob

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    y_cnn, keep_prob = deepnn(x)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels = y_, logits = y_cnn)

    cross_entropy = tf.reduce_mean(cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_cnn, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1500):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(
                        feed_dict = {x: batch[0],
                                    y_: batch[1],
                                    keep_prob: 1.0})
                print('step {}, training acc = {}'.format(i
                    , train_accuracy))
            train_step.run(feed_dict={x: batch[0],
                                      y_: batch[1],
                                      keep_prob: 0.5})

        test_acc = accuracy.eval(
                        feed_dict = {x: batch[0],
                                    y_: batch[1],
                                    keep_prob: 1.0})
        print('test acc = {}'.format(test_acc))
                                        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str,
                        default = '/tmp/tensorflow/mnist/input_data',
                        help = 'Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main = main, argv = [sys.argv[0]] + unparsed)
    
