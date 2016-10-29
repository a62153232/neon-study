# regression problem
from neon.data import ArrayIterator
import numpy as np
from neon.initializers import Gaussian
from neon.optimizers import GradientDescentMomentum
from neon.layers import Linear, Bias
from neon.layers import GeneralizedCost
from neon.transforms import SumSquared
from neon.models import Model
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser

parser = NeonArgparser(__doc__)
args = parser.parse_args()

X = np.random.randn(1000, 1)

y = 2 * X + 1 + 0.01 * np.random.randn(1000, 1)

train = ArrayIterator(X=X, y=y, make_onehot=False)
# Linear layer with one unit and a bias layer
init_norm = Gaussian(loc=0.0, scale=0.01)
layers = [Linear(1, init=init_norm), Bias(init=init_norm)]

mlp = Model(layers=layers)

# Loss function is the squared difference
cost = GeneralizedCost(costfunc=SumSquared())

# Learning rules
optimizer = GradientDescentMomentum(0.1, momentum_coef=0.9)

# run fit
mlp.fit(train, optimizer=optimizer, num_epochs=10, cost=cost,
        callbacks=Callbacks(mlp, **args.callback_args))

# print weights
slope = mlp.get_description(True)['model']['config']['layers'][0]['params']['W']
print "slope = ", slope
bias_weight = mlp.get_description(True)['model']['config']['layers'][1]['params']['W']
print "bias = ", bias_weight
