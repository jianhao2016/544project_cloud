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

import argparse
import os
import sys

import tensorflow as tf

import resnet_LBC

_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5

# We use a weight decay of 0.0002, which performs better than the 0.0001 that
# was originally suggested.
_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}


def record_dataset(filenames):
  """Returns an input pipeline Dataset from `filenames`."""
  record_bytes = _HEIGHT * _WIDTH * _DEPTH + 1
  return tf.data.FixedLengthRecordDataset(filenames, record_bytes)


def get_filenames(is_training, data_dir):
  """Returns a list of filenames."""
  data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

  assert os.path.exists(data_dir), (
      'Run cifar10_download_and_extract.py first to download and extract the '
      'CIFAR-10 data.')

  if is_training:
    return [
        os.path.join(data_dir, 'data_batch_%d.bin' % i)
        for i in range(1, _NUM_DATA_FILES + 1)
    ]
  else:
    return [os.path.join(data_dir, 'test_batch.bin')]


def parse_record(raw_record):
  """Parse CIFAR-10 image and label from a raw record."""
  # Every record consists of a label followed by the image, with a fixed number
  # of bytes for each.
  label_bytes = 1
  image_bytes = _HEIGHT * _WIDTH * _DEPTH
  record_bytes = label_bytes + image_bytes

  # Convert bytes to a vector of uint8 that is record_bytes long.
  record_vector = tf.decode_raw(raw_record, tf.uint8)

  # The first byte represents the label, which we convert from uint8 to int32
  # and then to one-hot.
  label = tf.cast(record_vector[0], tf.int32)
  label = tf.one_hot(label, _NUM_CLASSES)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      record_vector[label_bytes:record_bytes], [_DEPTH, _HEIGHT, _WIDTH])

  # Convert from [depth, height, width] to [height, width, depth], and cast as
  # float32.
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

  return image, label


def preprocess_image(image, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, _HEIGHT + 8, _WIDTH + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  return image


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  dataset = record_dataset(get_filenames(is_training, data_dir))

  if is_training:
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance. Because CIFAR-10
    # is a relatively small dataset, we choose to shuffle the full epoch.
    dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])

  dataset = dataset.map(parse_record)
  dataset = dataset.map(
      lambda image, label: (preprocess_image(image, is_training), label))

  dataset = dataset.prefetch(2 * batch_size)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)

  # Batch results by up to batch_size, and then fetch the tuple from the
  # iterator.
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()

  return images, labels


FLAGS = TEMP_opt()
params = {
        'depth': FLAGS.depth,
        'nClass': FLAGS.nClass,
        'kSize': FLAGS.convSize,
        'numChannels': FLAGS.numChannels,
        'units_in_FC': FLAGS.full,
        'data_format': FLAGS.data_format,
        'number_of_b': FLAGS.number_of_b,
        'sparsity': FLAGS.sparsity,
        'shared_weights': FLAGS.shared_weights,
        'batch_size': FLAGS.batch_size
        }

network = resnet_LBC.cifar10_resnet_LBC_generator(
        depth = params['depth'], nClass = params['nClass'],
        kSize = params['kSize'], numChannels = params['numChannels'],
        units_in_FC = params['units_in_FC'], data_format = params['data_format'],
        number_of_b = params['number_of_b'], sparsity = params['sparsity'],
        shared_weights = params['shared_weights'])

[features, labels] = input_fn(False, '../data/cifar10', 3)

inputs = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _DEPTH])
tf.summary.image('sample_images', inputs)

logits = network(inputs, False)

predictions = {
    'classes': tf.argmax(logits, axis=1),
    'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
}

cross_entropy = tf.losses.softmax_cross_entropy(
    logits=logits, onehot_labels=labels)

# Create a tensor named cross_entropy for logging purposes.
tf.identity(cross_entropy, name='cross_entropy')
tf.summary.scalar('cross_entropy', cross_entropy)

# Add weight decay to the loss.
loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
    [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
tf.identity(loss, name='loss')
tf.summary.scalar('loss', loss)

optimizer = tf.train.MomentumOptimizer(
    learning_rate=0.1,
    momentum=_MOMENTUM)


train_op = optimizer.minimize(loss)

# accuracy = tf.metrics.accuracy(
#     tf.argmax(labels, axis=1), predictions['classes'])
# tf.identity(accuracy[1], name = 'train_accuracy')
# tf.summary.scalar('train_accuracy', accuracy[1])

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('tensorboard_test/')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer.add_graph(sess.graph)
    summary = sess.run(merged)
    train_writer.add_summary(summary)
    
    for i in range(5):
        y = sess.run(labels)
        print(y)
        train_summary, _ = sess.run([train_op, merged]) 
        train_loss = sess.run(loss)
        print('iter {}, loss = {}, acc = {}'.format(i, train_loss, 0))
        [features, labels] = input_fn(True, '../data/cifar10', 5)
    # x = sess.run(inputs)
    # print(x.shape)
    # output_logit = sess.run(logits)
    # print(output_logit.shape)
        
          
          
          
          
