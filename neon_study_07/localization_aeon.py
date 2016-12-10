# -*- coding: utf-8 -*-

from aeon import DataLoader
from neon.initializers import Gaussian, GlorotUniform, Constant, Xavier

from neon.layers import GeneralizedCost, Affine, Sequential, MergeMultistream, Linear, Pooling, Conv, Dropout
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, RMSProp, Adam
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary, SumSquared, ObjectDetection,Identity
from neon.callbacks.callbacks import Callbacks
from neon.backends import gen_backend

# parser = NeonArgparser(__doc__)
# args = parser.parse_args(gen_be=False)

be = gen_backend(backend='gpu')

traindir = 'train'
imwidth = 256

image_config = dict(height=256, width=256, flip_enable=True)
localization_config = dict(class_names=["head"])
config = dict(type="image,localization",
              image=image_config,
              localization=localization_config,
              manifest_filename="train_json.csv",
              cache_directory="/home/auto-114/PycharmProjects/neon_study_10/cache",
              minibatch_size=1,
              macrobatch_size=1)

train = DataLoader(config, be)

# weight initialization
init_norm = Gaussian(loc=0.0, scale=0.01)

# setup model layers

init1_vgg = Xavier(local=True)
relu = Rectlin()

conv_params = {'strides': 1,
               'padding': 1,
               'init': init1_vgg,
               'bias': Constant(0),
               'activation': relu}

# Set up the model layers
layers = []

# set up 3x3 conv stacks with different feature map sizes
layers.append(Conv((3, 3, 64), name="skip", **conv_params))
layers.append(Conv((3, 3, 64), name="skip", **conv_params))
layers.append(Pooling(2, strides=2))
layers.append(Conv((3, 3, 128), name="skip", **conv_params))
layers.append(Conv((3, 3, 128), name="skip", **conv_params))
layers.append(Pooling(2, strides=2))
layers.append(Conv((3, 3, 256), **conv_params))
layers.append(Conv((3, 3, 256), **conv_params))
layers.append(Conv((3, 3, 256), **conv_params))
layers.append(Pooling(2, strides=2))
layers.append(Conv((3, 3, 512), **conv_params))
layers.append(Conv((3, 3, 512), **conv_params))
layers.append(Conv((3, 3, 512), **conv_params))
layers.append(Pooling(2, strides=2))
layers.append(Conv((3, 3, 512), **conv_params))
layers.append(Conv((3, 3, 512), **conv_params))
layers.append(Conv((3, 3, 512), **conv_params))

# not used after this layer
model = Model(layers=layers)

# cost = GeneralizedCost(costfunc=CrossEntropyBinary())
cost = GeneralizedCost(costfunc=SumSquared())
# fit and validate
optimizer = Adam(learning_rate=0.001)

# configure callbacks
# callbacks = Callbacks(model, eval_set=eval_set)
callbacks = Callbacks(model,train_set=train)

model.fit(train, cost=cost, optimizer=optimizer, num_epochs=10, callbacks=callbacks)
