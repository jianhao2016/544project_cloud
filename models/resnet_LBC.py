#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 qizai <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This file will run resnet with LBC module in cifar10.
Source code can be found on github.com/juefeix/lbcnn.torch
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from LBC_module import LBC
import tensorflow as tf
import numpy as np
from TEMP_opt import TEMP_opt

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

opt = TEMP_opt()
print('opt.batchSize = {}'.format(opt.batchSize))

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
        sparsity, padding, data_format):
    """
    inputs: a tensor
    nChIn/nChOut: number of input/output channels
    kW/kH : kernel size in LBC
    dW/dH : stride size in LBC
    # padW/padH: padding size
    sparsity: proportion of non zero elements in ancher weights
    padding : str, padding or not in RBC.
    """
    # ancher_shape = np.array([opt.convSize, opt.convSize,
    #                          opt.numChannels, opt.number_of_b])
    ancher_shape = np.array([kW, kH, nChIn, nChOut])
    num_elements = np.product(ancher_shape, dtype = int)
    num_non_zero = num_elements * sparsity
    num_non_zero = num_non_zero.astype(int)

    #initialize ancher weights
    ancher_weights = np.zeros(shape = ancher_shape, dtype = np.float)
    ancher_weights = np.reshape(ancher_weights, newshape = [num_elements])
    if opt.shared_weights:
        np.random.seed(42)
    index_non_zero = np.random.choice(num_elements, num_non_zero, replace = False)
    for i in index_non_zero:
        ancher_weights[i] = np.random.binomial(1, 0.5) * 2 - 1
    ancher_weights = np.reshape(ancher_weights, newshape = ancher_shape)
    ancher_weights_tensor = tf.constant(ancher_weights, dtype = tf.float32)

    if opt.data_format == 'channels_first':
        tf_format = 'NCHW'
        tf_strides = [1, 1, dW, dH]
    else:
        tf_format = 'NHWC'
        tf_strides = [1, dW, dH, 1]
    
    diff_map = tf.nn.conv2d(inputs, filter = ancher_weights_tensor,
                            strides = tf_strides, padding = padding,
                            data_format = data_format)
    return diff_map
    

# def basic_block(inputs, is_training, n_ChIn, n_ChOut, filter_size, strides, 
#         data_format, number_of_b, sparsity, shared_weights, name):
def basic_block(inputs, nChIn, nChTmp, kSize, sparsity, is_training, 
        data_format, block_name):
    """
    basic resnet block, with LBC module replacement.
    input channels -> nChIn
    intermediate channels -> nChTmp, which is just number_of_b
    output channels -> nChIn
    """
    with tf.name_scope(block_name):
        shortcut = inputs
        with tf.name_scope('batch_normalization'):
            inputs = batch_norm_relu(inputs, is_training, data_format)
        with tf.name_scope('random_binary_conv'):
            inputs = random_binary_convlution(inputs, nChIn = nChIn, nChOut = nChTmp, 
                    kW = kSize, kH = kSize, dW = 1, dH = 1, sparsity = sparsity,
                    padding = 'SAME', data_format = data_format)

        inputs = tf.nn.relu(inputs)
        with tf.name_scope('1x1 conv'):
            # the second conv doesn't need any padding, since it's 1x1.
            inputs = tf.layers.conv2d(inputs = inputs, filters = nChIn,
                                    kernel_size = [1, 1],
                                    paddind = 'valid',
                                    data_format = data_format,
                                    use_bias = False)
        output = shortcut + inputs
    return output

def cifar10_resnet_LBC_generator(opt):
    """
    returns a model function that takes inputs and is_training and compute the output
    tensor.
    """
    kSize = opt.convSize
    nInputPlane = opt.nInputPlane
    nChIn = opt.numChannels
    nChTmp = opt.number_of_b
    sparsity = opt.sparsity
    data_format = opt.data_format
    shared_weights = opt.shared_weights
    depth = opt.depth
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
        
        for i in range(1, depth):
            block_name = 'residual block ' + str(i)
            inputs = basic_block(inputs, nChIn, nChTmp, kSize, sparsity, is_training,
                    data_format, block_name = block_name)

        # inputs = batch_norm_relu(inputs, is_training, data_format)
        inputs = tf.layers.average_pooling2d(inputs, pool_size = 5,
                strides = 5, padding = 'valid', data_format = data_format)
        inputs = tf.identity(inputs, name = 'final_avg_pool')
        
        # a two layer FC network with dropout while training.
        inputs = tf.reshape(inputs, [-1, shape_after_avg])
        inputs = tf.layers.dropout(inputs, training = is_training)
        inputs = tf.layers.dense(inputs, units = opt.full, activation = tf.nn.relu)
        inputs = tf.layers.dropout(inputs, training = is_training)
        inputs = tf.layers.dense(inputs, units = opt.nClass)
        inputs = tf.identity(inputs, name = 'final_dense_out')
        return inputs

    return model

cifar10_resnet_LBC_generator(opt)
