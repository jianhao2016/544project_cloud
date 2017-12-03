#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 qizai <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This is a naive training resnet.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from TEMP_opt import TEMP_opt

import numpy as np
import tensorflow as tf
import resnet_LBC

opt = TEMP_opt()

_image_width = 32
_image_height = 32
_channels = 3
_train_dataset_size = 50000
_test_dataset_size = 10000
_WEIGHT_DECAY = 2e-4
_momentum = 0.9

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# unpacking training and test data
b1 = unpickle('~/projects/544project/data/cifar-10-batches-py/data_batch_1')
b2 = unpickle('~/projects/544project/data/cifar-10-batches-py/data_batch_2')
b3 = unpickle('~/projects/544project/data/cifar-10-batches-py/data_batch_3')
b4 = unpickle('~/projects/544project/data/cifar-10-batches-py/data_batch_4')
b5 = unpickle('~/projects/544project/data/cifar-10-batches-py/data_batch_5')

test = unpickle('~/projects/544project/data/cifar-10-batches-py/test-batch')

# preparing test data
test_data = test['data']
test_label = test['label']

# preparing training data
train_data = np.concatenate([b1['data'],b2['data'],b3['data'],b4['data'],b5['data']],axis=0)
train_label = np.concatenate([b1['labels'],b2['labels'],b3['labels'],b4['labels'],b5['labels']],axis=0)

#Reshaping data
# if opt.data_format == 'channels_first':
#     train_data = np.reshape(train_data, newshape = 
#         [_train_dataset_size, _channels, _image_height, _image_width])
#     test_data = np.reshape(test_data, newshape = 
#         [_test_dataset_size, _channels, _image_height, _image_width])
#     
# elif opt.data_format == 'channels_last':
#     train_data = np.reshape(train_data, newshape = 
#         [_train_dataset_size, _image_height, _image_width, _channels])
#     test_data = np.reshape(test_data, newshape = 
#         [_test_dataset_size, _image_height, _image_width, _channels])
# else:
#     print('data_format error. check opt file')
#     exit(1)
train_data = np.reshape(train_data, newshape = 
    [_train_dataset_size, _image_height, _image_width, _channels])
test_data = np.reshape(test_data, newshape = 
    [_test_dataset_size, _image_height, _image_width, _channels])


network = resnet_LBC.cifar10_resnet_LBC_generator(depth = opt.depth,
        nClass = opt.nClass, kSize = opt.kSize, numChannels = opt.numChannels,
        units_in_FC = opt.full, data_format = opt.data_format,
        number_of_b = opt.number_of_b, sparsity = opt.sparsity,
        shared_weights = opt.shared_weights)

# construct the graph
images = tf.placeholder(tf.float32, shape = [None, _image_height, _image_width, _channels])
labels = tf.placeholder(tf.float32, shape = [None])
one_hot_labels = tf.one_hot(labels, depth = opt.nClass)
learning_rate = tf.placeholder(tf.float32, shape = [])

# compute the output of a graph and see the loss/accuracy
logits = network(images, is_training = True)
cross_entropy = tf.losses.softmax_cross_entropy(one_hot_labels, logits)

loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
    [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = _momentum)
train_op = optimizer.minimize(loss)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(opt.nEpochs):
        # shuffle the data set
        for iter in range(_train_dataset_size // opt.batch_size):

