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
# from resnet_LBC import random_binary_convlution
import base64

opt = TEMP_opt()
# opt.depth = 20

_image_width = 32
_image_height = 32
_channels = 3
_train_dataset_size = 50000
_test_dataset_size = 10000
_WEIGHT_DECAY = 2e-4
_momentum = 0.9

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def shuffleDataSet(images, labels):
    assert len(images) == len(labels)
    p = np.random.permutation(len(images))
    return images[p], labels[p]
    
# unpacking training and test data
b1 = unpickle('../data/cifar-10-batches-py/data_batch_1')
b2 = unpickle('../data/cifar-10-batches-py/data_batch_2')
b3 = unpickle('../data/cifar-10-batches-py/data_batch_3')
b4 = unpickle('../data/cifar-10-batches-py/data_batch_4')
b5 = unpickle('../data/cifar-10-batches-py/data_batch_5')

test = unpickle('../data/cifar-10-batches-py/test_batch')
for key, _ in test.items():
    print(repr(key), type(key))
for key, _ in b1.items():
    print(repr(key), type(key))

# preparing test data
test_data = test[b'data']
test_label = test[b'labels']

# preparing training data
train_data = np.concatenate([b1[b'data'],b2[b'data'],b3[b'data'],b4[b'data'],b5[b'data']],axis=0)
train_label = np.concatenate([b1[b'labels'],b2[b'labels'],b3[b'labels'],b4[b'labels'],b5[b'labels']],axis=0)

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
    [-1, _channels, _image_height, _image_width])
test_data = np.reshape(test_data, newshape = 
    [-1, _channels, _image_height, _image_width])
train_data = np.array(train_data, dtype=float) / 255.0
test_data = np.array(test_data, dtype=float) /255.0
train_data = train_data.transpose([0, 3, 2, 1])
test_data = test_data.transpose([0, 3, 2, 1])


# network = resnet_LBC.cifar10_resnet_vanilla_generator(depth = opt.depth,
#         nClass = opt.nClass, kSize = opt.convSize, numChannels = opt.numChannels,
#         units_in_FC = opt.full, data_format = opt.data_format,
#         number_of_b = opt.number_of_b, sparsity = opt.sparsity,
#         shared_weights = opt.shared_weights)

network = resnet_LBC.cifar10_resnet_LBC_generator(depth = opt.depth,
        nClass = opt.nClass, kSize = opt.convSize, numChannels = opt.numChannels,
        units_in_FC = opt.full, data_format = opt.data_format,
        number_of_b = opt.number_of_b, sparsity = opt.sparsity,
        shared_weights = opt.shared_weights)

# def LBC_resnet(inputs, is_training, opt):
#     def basic_res_block(inputs, nChIn, nChTmp, kSize, is_training, data_format, 
#             sparsity, shared_weights, block_name):
#         output = tf.layers.batch_normalization(inputs = inputs, 

# construct the graph
images = tf.placeholder(tf.float32, shape = [None, _image_height, _image_width, _channels])
labels = tf.placeholder(tf.float32, shape = [None])
one_hot_labels = tf.one_hot(tf.cast(labels, tf.int32), depth = opt.nClass)
learning_rate = tf.placeholder(tf.float32, shape = [])
training_rate = 0.1
is_training = tf.placeholder(tf.bool, shape = [], name = 'training_flag')
# we can probably add drop-out rate here. As a placeholder.

# compute the output of a graph and see the loss/accuracy
logits = network(images, is_training = is_training)
cross_entropy = tf.losses.softmax_cross_entropy(one_hot_labels, logits)

loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
   [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
#loss = cross_entropy

optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = _momentum)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    total_step = opt.nEpochs * _train_dataset_size // opt.batch_size
    for epoch in range(opt.nEpochs):
        # shuffle the data set
        train_data, train_label = shuffleDataSet(train_data, train_label)
        for iter in range(_train_dataset_size // opt.batch_size):
            # decrease the learning rate in a naive way.
            step = epoch * (_train_dataset_size//opt.batch_size) + iter
            if step in [total_step//4, total_step//2, total_step * 3//4, total_step * 7 // 8]:
                training_rate *= 0.1
            # sample batchs from training data.
            images_batch = train_data[iter : iter + opt.batch_size]
            labels_batch = train_label[iter : iter + opt.batch_size]
            feed_dict_1 = {images : images_batch,
                         labels : labels_batch,
                         learning_rate : training_rate,
                         is_training : True}
            sess.run(train_op, feed_dict = feed_dict_1)
            train_loss_w_bn, train_xentro_w_bn, train_acc_w_bn = sess.run(
                    [loss, cross_entropy, accuracy], feed_dict = feed_dict_1)
            feed_dict_2 = {images : images_batch,
                         labels : labels_batch,
                         learning_rate : training_rate,
                         is_training : False}
            train_loss_wo_bn, train_xentro_wo_bn, train_acc_wo_bn = sess.run(
                    [loss, cross_entropy, accuracy], feed_dict = feed_dict_2)
            if iter%50 == 0:
                print('learning_rate', training_rate)
                print('step {}, with batch norm training loss = {}, cross_entropy = {}, training accuracy = {}'.format(
                    step, train_loss_w_bn, train_xentro_w_bn, train_acc_w_bn))
                print('step {}, without batch norm training loss = {}, cross_entropy = {}, training accuracy = {}'.format(
                    step, train_loss_wo_bn, train_xentro_wo_bn, train_acc_wo_bn))
                # val_loss, val_xe, val_acc = sess.run([loss, cross_entropy, accuracy],
                #         feed_dict = feed_dict_1)
                # print('validatation, step = {}, loss = {}, xe = {}, acc = {}'.format(
                #         step, val_loss, val_xe, val_acc))
                print('----')
        # do evaluation every epoch.
        eval_loss = 0
        eval_acc = 0
        for i in range(_test_dataset_size//opt.batch_size):
            eval_images = test_data
            eval_labels = test_label
            test_dict = {images : eval_images[i: i + opt.batch_size],
                         labels : eval_labels[i: i + opt.batch_size],
                         is_training: False}
            test_batch_loss, test_batch_acc = sess.run([loss, accuracy], feed_dict = test_dict)
            eval_loss += test_batch_loss
            eval_acc += test_batch_acc
        eval_loss = eval_loss/(_test_dataset_size//opt.batch_size)
        eval_acc = eval_acc/(_test_dataset_size//opt.batch_size)
        print('epoch# {}, evaluation loss = {}, accuracy = {}'.format(
            epoch, eval_loss, eval_acc))

