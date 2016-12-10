#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 10:40:55 2016

@author: auto-114

Convolutional autoencoder example network on MNIST data set.

Usage:
    python examples/conv_autoencoder.py
"""

import numpy as np
from neon import logger as neon_logger
from neon.data import ArrayIterator, MNIST
from neon.initializers import Uniform
from neon.layers import Conv, Pooling, GeneralizedCost, Deconv
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, SumSquared
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

# Load dataset
dataset = MNIST(path=args.data_dir)
(X_train, y_train), (X_test, y_test), nclass = dataset.load_data()

# Set input and target to X_train
train = ArrayIterator(X_train, lshape=(1, 28, 28))

# Initialize the weights and the learning rule
init_uni = Uniform(low=-0.1, high=0.1)
opt_gdm = GradientDescentMomentum(learning_rate=0.001, momentum_coef=0.9)

# Strided conv autoencoder
bn = False
layers = [Conv((4, 4, 8), init=init_uni, activation=Rectlin(), batch_norm=bn),
          Pooling(2),
          Conv((4, 4, 32), init=init_uni, activation=Rectlin(), batch_norm=bn),
          Pooling(2),
          Deconv(fshape=(4, 4, 8), init=init_uni,
                 activation=Rectlin(), batch_norm=bn),
          Deconv(fshape=(3, 3, 8), init=init_uni,
                 activation=Rectlin(), strides=2, batch_norm=bn),
          Deconv(fshape=(2, 2, 1), init=init_uni, strides=2, padding=1)]

# Define the cost
cost = GeneralizedCost(costfunc=SumSquared())

model = Model(layers=layers)

# configure callbacks
callbacks = Callbacks(model, **args.callback_args)

# Fit the model
model.fit(train, optimizer=opt_gdm, num_epochs=args.epochs,
          cost=cost, callbacks=callbacks)

# Plot the reconstructed digits
try:
    from matplotlib import pyplot, cm
    fi = 0
    nrows = 10
    ncols = 12
    test = np.zeros((28 * nrows, 28 * ncols))
    idxs = [(row, col) for row in range(nrows) for col in range(ncols)]
    for row, col in idxs:
        im = model.layers.layers[-1].outputs.get()[:, fi].reshape((28, 28))
        test[28 * row:28 * (row + 1):, 28 * col:28 * (col + 1)] = im
        fi = fi + 1
    pyplot.matshow(test, cmap=cm.gray)
    pyplot.savefig('Reconstructed.png')
except ImportError:
    neon_logger.display(
'matplotlib needs to be manually installed to generate plots')
