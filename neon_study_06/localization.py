#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 15:35:53 2016

@author: auto-114
"""
import numpy as np
import h5py
from neon.data import ArrayIterator
from neon.initializers import Gaussian
from neon.layers import GeneralizedCost, Affine, Sequential, MergeMultistream, Linear, Pooling, Conv
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, RMSProp
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary, SumSquared, ObjectDetection
from neon.callbacks.callbacks import Callbacks
from neon.backends import gen_backend


# parser = NeonArgparser(__doc__)
# args = parser.parse_args(gen_be=False)

be = gen_backend(batch_size=100, backend='gpu')

traindir = 'train'
imwidth = 256
data = np.load("data.npy")
coordinate = np.load("coordinate.npy")

# X_train, y_train = data[0:1], coordinate[0:256]
X_train, y_train = np.asarray(data[0:3000]), np.asarray(coordinate[0:3000])
train_set = ArrayIterator(X_train[0:2000], y_train[0:2000], make_onehot=False, lshape=(3, 256, 256))
eval_set = ArrayIterator(X_train[1000:2000], y_train[1000:2000], make_onehot=False, lshape=(3, 256, 256))
test_set = ArrayIterator(X_train[2000:2500], y_train[2000:2500], make_onehot=False, lshape=(3, 256, 256))

# weight initialization
init_norm = Gaussian(loc=0.0, scale=0.01)

# setup model layers

layers = [Conv((5, 5, 16), init=init_norm, activation=Rectlin()),
          Pooling(2),
          Conv((5, 5, 32), init=init_norm, activation=Rectlin()),
          Pooling(2),
          Conv((3, 3, 32), init=init_norm, activation=Rectlin()),
          Pooling(2),
          Affine(nout=100, init=init_norm, activation=Rectlin()),
          Linear(nout=4, init=init_norm)]

model = Model(layers=layers)

# cost = GeneralizedCost(costfunc=CrossEntropyBinary())
cost = GeneralizedCost(costfunc=SumSquared())
# fit and validate
optimizer = RMSProp()

# configure callbacks
callbacks = Callbacks(model, eval_set=eval_set, eval_freq=1)

model.fit(train_set, cost=cost, optimizer=optimizer, num_epochs=10, callbacks=callbacks)
y_test = model.get_outputs(test_set)
