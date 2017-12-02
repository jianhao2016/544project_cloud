#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 qizai <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This function will run the LBC resnet on cifar10.
"""
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
        self.nEpochs = 0 # number of total epochs to run
        self.epochNumber = 1 # manual epoch number
        self.batchSize = 20
        self.data_format = None


