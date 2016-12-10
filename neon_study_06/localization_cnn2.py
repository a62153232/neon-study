#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 20:57:51 2016

@author: auto-114
"""
import numpy as np
import h5py
from neon.data import ArrayIterator,HDF5Iterator
from neon.initializers import Gaussian,GlorotUniform,Constant, Xavier

from neon.layers import GeneralizedCost, Affine, Sequential, MergeMultistream, Linear, Pooling, Conv,Dropout
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, RMSProp, Adam
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary, SumSquared, ObjectDetection
from neon.callbacks.callbacks import Callbacks
from neon.backends import gen_backend

# parser = NeonArgparser(__doc__)
# args = parser.parse_args(gen_be=False)

be = gen_backend(batch_size=128, backend='gpu')

traindir = 'train'
imwidth = 256

train_set = HDF5Iterator('whale_train.h5')
eval_set = HDF5Iterator('whale_eval.h5')
test_set = HDF5Iterator('whale_test.h5')

# weight initialization
init_norm = Gaussian(loc=0.0, scale=0.01)

# setup model layers
               
layers = [Conv((7, 7, 16), init=init_norm, activation=Rectlin()),
          Pooling((2, 2)),  
          
          Conv((3, 3, 32), init=init_norm, activation=Rectlin()),
          Conv((3, 3, 32), init=init_norm, activation=Rectlin()),
          Conv((3, 3, 32), init=init_norm, activation=Rectlin()),
          
          Conv((3, 3, 64), strides=2, padding=1, init=init_norm, activation=Rectlin()),
          Conv((3, 3, 64), init=init_norm, activation=Rectlin()),
          Conv((3, 3, 64), init=init_norm, activation=Rectlin()),
          
          Conv((3, 3, 128), strides=2, padding=1, init=init_norm, activation=Rectlin()),
          Conv((3, 3, 128), init=init_norm, activation=Rectlin()),
          Conv((3, 3, 128), init=init_norm, activation=Rectlin()),
          
          Conv((3, 3, 256), strides=2, padding=1, init=init_norm, activation=Rectlin()),
          Conv((3, 3, 256), init=init_norm, activation=Rectlin()),
          Conv((3, 3, 256), init=init_norm, activation=Rectlin()),
          Pooling((8, 8)),
          Dropout(0.5),
          Linear(nout=4, init=Gaussian(loc=0.0, scale=0.01))]


model = Model(layers=layers)

# cost = GeneralizedCost(costfunc=CrossEntropyBinary())
cost = GeneralizedCost(costfunc=SumSquared())
# fit and validate
optimizer = Adam(learning_rate=0.001)

# configure callbacks
callbacks = Callbacks(model, eval_set=eval_set)

model.fit(train_set, cost=cost, optimizer=optimizer, num_epochs=50, callbacks=callbacks)
y_test = model.get_outputs(test_set)
