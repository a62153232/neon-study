# -*- coding: utf-8 -*-

import numpy as np
import h5py
from neon.data import ArrayIterator,HDF5Iterator
from neon.initializers import Gaussian,GlorotUniform,Constant, Xavier

from neon.layers import GeneralizedCost, Affine, Sequential, MergeMultistream, Linear, Pooling, Conv,Dropout
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, RMSProp
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary, SumSquared, ObjectDetection
from neon.callbacks.callbacks import Callbacks
from neon.backends import gen_backend

# parser = NeonArgparser(__doc__)
# args = parser.parse_args(gen_be=False)

be = gen_backend(batch_size=2, backend='gpu')

traindir = 'train'
imwidth = 256

train_set = HDF5Iterator('whale_train.h5')
eval_set = HDF5Iterator('whale_eval.h5')
test_set = HDF5Iterator('whale_test.h5')

# weight initialization
init_norm = Gaussian(loc=0.0, scale=0.01)

# setup model layers
relu = Rectlin()
conv_params = {'strides': 1,
               'padding': 1,
               'init': Xavier(local=True),
               'bias': Constant(0),
               'activation': relu}
               
vgg_layers = []

# set up 3x3 conv stacks with different number of filters
vgg_layers.append(Conv((3, 3, 64), **conv_params))
vgg_layers.append(Conv((3, 3, 64), **conv_params))
vgg_layers.append(Pooling(2, strides=2))
vgg_layers.append(Conv((3, 3, 128), **conv_params))
vgg_layers.append(Conv((3, 3, 128), **conv_params))
vgg_layers.append(Pooling(2, strides=2))
vgg_layers.append(Conv((3, 3, 256), **conv_params))
vgg_layers.append(Conv((3, 3, 256), **conv_params))
vgg_layers.append(Conv((3, 3, 256), **conv_params))
vgg_layers.append(Pooling(2, strides=2))
vgg_layers.append(Conv((3, 3, 512), **conv_params))
vgg_layers.append(Conv((3, 3, 512), **conv_params))
vgg_layers.append(Conv((3, 3, 512), **conv_params))
vgg_layers.append(Pooling(2, strides=2))
vgg_layers.append(Conv((3, 3, 512), **conv_params))
vgg_layers.append(Conv((3, 3, 512), **conv_params))
vgg_layers.append(Conv((3, 3, 512), **conv_params))
vgg_layers.append(Pooling(2, strides=2))
vgg_layers.append(Affine(nout=4096, init=GlorotUniform(), bias=Constant(0), activation=relu))
vgg_layers.append(Dropout(keep=0.5))
vgg_layers.append(Affine(nout=4096, init=GlorotUniform(), bias=Constant(0), activation=relu))
vgg_layers.append(Dropout(keep=0.5))

vgg_layers.append(Linear(nout=4, init=GlorotUniform()))

model = Model(layers=vgg_layers)

# cost = GeneralizedCost(costfunc=CrossEntropyBinary())
cost = GeneralizedCost(costfunc=SumSquared())
# fit and validate
optimizer = RMSProp()

# configure callbacks
callbacks = Callbacks(model, eval_set=eval_set)

model.fit(train_set, cost=cost, optimizer=optimizer, num_epochs=10, callbacks=callbacks)
y_test = model.get_outputs(test_set)

