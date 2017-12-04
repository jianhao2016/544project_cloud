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


class TEMP_opt:
    def __init__(self):
        self.nClass = 10
        self.stride = 1
        self.sparsity = 0.9
        self.nInputPlane = 3
        self.numChannels = 128 # number of intermediate layers between blocks, i.e. nChIn
        self.number_of_b = 512 # number of binary filters in LBC, i.e. nChTmp
        self.full = 512 # number of hidden units in FC
        self.convSize = 3 # LB convolutional filter size
        self.depth = 20 # number of blocks
        self.weightDecay = 1e-4
        self.LR = 1e-4 #initial learning rate
        self.nEpochs = 100 # number of total epochs to run
        # self.epochNumber = 1 # manual epoch number
        self.batch_size = 128
        self.data_format = None
        self.shared_weights = False


import numpy as np
import tensorflow as tf
# import resnet_LBC
# these two params are use in batch norm
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

# opt = TEMP_opt()
# print('opt.batchSize = {}'.format(opt.batchSize))

def batch_norm_relu(inputs, is_training, data_format):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=is_training, fused=True)
  inputs = tf.nn.relu(inputs)
  return inputs

def random_binary_convlution(inputs, nChIn, nChOut, kW, kH, dW, dH, 
        padding, data_format, sparsity, shared_weights):
    """
    inputs: a tensor
    nChIn/nChOut: number of input/output channels
    kW/kH : kernel size in LBC
    dW/dH : stride size in LBC
    padding : str, padding or not in RBC.
    # padW/padH: padding size
    sparsity: proportion of non zero elements in ancher weights
    shared_weights: boolean, whether uses same binary filter or not.
    """
    ancher_shape = np.array([kW, kH, nChIn, nChOut])
    num_elements = np.product(ancher_shape, dtype = int)
    num_non_zero = num_elements * sparsity
    num_non_zero = num_non_zero.astype(int)

    #initialize ancher weights
    ancher_weights = np.zeros(shape = ancher_shape, dtype = np.float)
    ancher_weights = np.reshape(ancher_weights, newshape = [num_elements])
    if shared_weights:
        np.random.seed(42)
    index_non_zero = np.random.choice(num_elements, num_non_zero, replace = False)
    for i in index_non_zero:
        ancher_weights[i] = np.random.binomial(1, 0.5) * 2 - 1
    ancher_weights = np.reshape(ancher_weights, newshape = ancher_shape)
    ancher_weights_tensor = tf.constant(ancher_weights, dtype = tf.float32)

    if data_format == 'channels_first':
        tf_format = 'NCHW'
        tf_strides = [1, 1, dW, dH]
    else:
        tf_format = 'NHWC'
        tf_strides = [1, dW, dH, 1]
    
    diff_map = tf.nn.conv2d(inputs, filter = ancher_weights_tensor,
                            strides = tf_strides, padding = padding,
                            data_format = tf_format)
    return diff_map
    

def basic_block_LBC(inputs, nChIn, nChTmp, kSize, is_training, data_format, 
        sparsity, shared_weights, block_name):
    """
    basic resnet block, with LBC module replacement.
    nChIn : number of input channels to the block. Notice that the in/out of a
            block has the same 'depth'/number of channels
    nChTmp: number of binary filters used in the block. Cancel out by the 
            second conv in block.
    kSize:  filter size in RBC.
    is_training: a boolean, tell if training
    data_format: 'channels_first' or 'channels_last'
    sparsity/shared_weights: params used in RBC.
    block_name: string. name of block.
    """
    with tf.name_scope(block_name):
        shortcut = inputs
        with tf.name_scope('batch_normalization'):
            inputs = batch_norm_relu(inputs, is_training, data_format)
        with tf.name_scope('random_binary_conv'):
            inputs = random_binary_convlution(inputs, nChIn = nChIn, nChOut = nChTmp,
                    kW = kSize, kH = kSize, dW = 1, dH = 1, padding = 'SAME',
                    data_format = data_format, sparsity = sparsity,
                    shared_weights = shared_weights)

        inputs = tf.nn.relu(inputs)
        with tf.name_scope('1x1_conv'):
            # the second conv doesn't need any padding, since it's 1x1.
            inputs = tf.layers.conv2d(inputs = inputs, filters = nChIn,
                                    kernel_size = [1, 1],
                                    padding = 'valid',
                                    data_format = data_format,
                                    use_bias = False)
        output = shortcut + inputs
    return output

def cifar10_resnet_LBC_generator(depth, nClass, kSize, numChannels, 
        units_in_FC, data_format, number_of_b, sparsity, shared_weights):
    """
    depth: how many blocks to use in resnet.
    nClass: how many classes in the output layer
    kSize: convolution size in resnet.
    numChannels: how many filters to use in the first conv layer.
                 i.e. number of input channels to the blocks chain.
    units_in_FC: number of units in the first fully connected layer.
    data_format: 'channels_first' or 'channels_last'
    number_of_b: number of binary filters. i.e. the filters used in RBC
    sparsity/shared_weights: params used in LBC
    returns a model function that takes inputs and is_training and compute the output
    tensor.
    """
    nChIn = numChannels
    nChTmp = number_of_b
    # after a 5x5 non-overlapping average pooling, the cifar10 image origin
    # size is 32x32, and now is only 6x6 left.
    shape_after_avg = 6 * 6
    
    if data_format is None:
        data_format = (
                'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    def model(inputs, is_training):
        """
        Constructs the ResNet model given the inputs.
        """
        if data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        inputs = tf.layers.conv2d(inputs = inputs, filters = nChIn,
                kernel_size = [kSize, kSize], strides = (1, 1),
                padding = 'SAME', data_format = data_format)
        # not necessary to add batch normalization, since each basic block hase bn
        # inputs = batch_norm_relu(inputs, is_training, data_format)
        
        for i in range(depth):
            block_name = 'LBC_residual_block_' + str(i)
            inputs = basic_block_LBC(inputs, nChIn, nChTmp, kSize, is_training,
                    data_format = data_format, sparsity = sparsity,
                    shared_weights = shared_weights, block_name = block_name)

        inputs = batch_norm_relu(inputs, is_training, data_format)
        inputs = tf.layers.average_pooling2d(inputs, pool_size = 5,
                strides = 5, padding = 'valid', data_format = data_format)
        inputs = tf.identity(inputs, name = 'final_avg_pool')
        
        # a two layer FC network with dropout while training.
        inputs = tf.reshape(inputs, [-1, numChannels * shape_after_avg])
        inputs = tf.layers.dropout(inputs, training = is_training)
        inputs = tf.layers.dense(inputs, units = units_in_FC, activation = tf.nn.relu)
        inputs = tf.layers.dropout(inputs, training = is_training)
        inputs = tf.layers.dense(inputs, units = nClass)
        inputs = tf.identity(inputs, name = 'final_dense_out')
        return inputs

    return model


# from resnet_LBC import random_binary_convlution
import base64
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type = str,
                    help='The directory to the stored cifar10 data python format')

parser.add_argument('--depth', type = int, default = 15,
                    help='The depth of resnet')

parser.add_argument('--result_dir', type = str,
                    help='the directory to save the training result')

FLAGS, unparsed = parser.parse_known_args()

opt = TEMP_opt()
opt.depth = FLAGS.depth

path2Data = FLAGS.data_dir

output_result_file = open('{}resnet_LBC_result_output_file'.format(FLAGS.result_dir), 'w')

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

output_result_file.write('extracting data from: {}cifar-10-batches-py/'.format(path2Data))
output_result_file.write('\n')
# unpacking training and test data
b1 = unpickle(path2Data + 'cifar-10-batches-py/data_batch_1')
b2 = unpickle(path2Data + 'cifar-10-batches-py/data_batch_2')
b3 = unpickle(path2Data + 'cifar-10-batches-py/data_batch_3')
b4 = unpickle(path2Data + 'cifar-10-batches-py/data_batch_4')
b5 = unpickle(path2Data + 'cifar-10-batches-py/data_batch_5')

test = unpickle(path2Data + 'cifar-10-batches-py/test_batch')
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
output_result_file.write('depth of resnet = {}'.format(opt.depth))
output_result_file.write('\n')
network = cifar10_resnet_LBC_generator(depth = opt.depth,
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
                output_result_file.write('learning_rate = {}'.format(training_rate))
                output_result_file.write('\n')

                output_result_file.write('step {}, with batch norm training loss = {}, cross_entropy = {}, training accuracy = {}'.format(
                    step, train_loss_w_bn, train_xentro_w_bn, train_acc_w_bn))
                output_result_file.write('\n')

                output_result_file.write('step {}, without batch norm training loss = {}, cross_entropy = {}, training accuracy = {}'.format(
                    step, train_loss_wo_bn, train_xentro_wo_bn, train_acc_wo_bn))
                output_result_file.write('\n')

                # val_loss, val_xe, val_acc = sess.run([loss, cross_entropy, accuracy],
                #         feed_dict = feed_dict_1)
                # output_result_file.write('validatation, step = {}, loss = {}, xe = {}, acc = {}'.format(
                #         step, val_loss, val_xe, val_acc))
                output_result_file.write('----')
                output_result_file.write('\n')
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
        output_result_file.write('epoch# {}, evaluation loss = {}, accuracy = {}'.format(
            epoch, eval_loss, eval_acc))
        output_result_file.write('\n')

output_result_file.close()
