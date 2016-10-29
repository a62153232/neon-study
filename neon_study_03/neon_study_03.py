import cPickle
import os
import gzip
import numpy as np
from neon.callbacks.callbacks import Callbacks

from neon.initializers import Gaussian
from neon.layers import GeneralizedCost, Affine, Conv, Pooling, Dropout
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, CrossEntropyBinary, Misclassification, Softmax
from neon.util.argparser import NeonArgparser
from neon import logger as neon_logger
from neon.data import ArrayIterator


def load_data():
    # Download the MNIST dataset if it is not present
    if (not os.path.isdir('data')):
        os.mkdir('data')  # make directory 'data'
    dataset = 'data/mnist.pkl.gz'
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib

        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    train_set_x, train_set_y = list(train_set)
    valid_set_x, valid_set_y = list(valid_set)
    test_set_x, test_set_y = list(test_set)

    train_set_x = np.row_stack((train_set_x, test_set_x))
    print train_set_x.shape
    train_set_y = np.append(train_set_y, test_set_y)
    print train_set_y.shape

    return train_set_x, train_set_y, valid_set_x, valid_set_y


parser = NeonArgparser(__doc__)
args = parser.parse_args()
args.epochs = 30

train_set_x, train_set_y, valid_set_x, valid_set_y = load_data()
train_set = ArrayIterator(train_set_x, train_set_y, nclass=10, lshape=(1, 28, 28))
valid_set = ArrayIterator(valid_set_x, valid_set_y, nclass=10, lshape=(1, 28, 28))
init_uni = Gaussian(loc=0.0, scale=0.01)

# setup model layers
layers = [Conv(fshape=(5, 5, 32), init=init_uni, activation=Rectlin()),
          Pooling(fshape=2, strides=2),
          Conv(fshape=(5, 5, 32), init=init_uni, activation=Rectlin()),
          Pooling(fshape=2, strides=2),
          Dropout(),
          Affine(nout=500, init=init_uni, activation=Rectlin()),
          Dropout(),
          Affine(nout=10, init=init_uni, activation=Softmax())]

# setup cost function as CrossEntropy
cost = GeneralizedCost(costfunc=CrossEntropyBinary())

# setup optimizer
optimizer = GradientDescentMomentum(
    0.1, momentum_coef=0.9, stochastic_round=args.rounding)

# initialize model object
mlp = Model(layers=layers)

# configure callbacks
callbacks = Callbacks(mlp, eval_set=valid_set, **args.callback_args)

# run fit
mlp.fit(train_set, optimizer=optimizer,
        num_epochs=args.epochs, cost=cost, callbacks=callbacks)
error_rate = mlp.eval(valid_set, metric=Misclassification())
neon_logger.display('Misclassification error = %.1f%%' % (error_rate * 100))
