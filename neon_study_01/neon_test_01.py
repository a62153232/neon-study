from neon.initializers import Gaussian
from neon.layers import Affine, Conv, Pooling, Dropout
from neon.transforms import Rectlin, Softmax
from neon.models import Model
from neon.layers import GeneralizedCost
from neon.transforms import CrossEntropyMulti
from neon.optimizers import GradientDescentMomentum
from neon.callbacks.callbacks import Callbacks
from neon.data import MNIST
from neon.util.argparser import NeonArgparser
from neon.initializers import Uniform
from neon.transforms import Misclassification

parser = NeonArgparser(__doc__)
args = parser.parse_args()

MNIST_dataset = MNIST()
train_set = MNIST_dataset.train_iter
test_set = MNIST_dataset.valid_iter

init_uni = Uniform(low=-0.1, high=0.1)
layers = [Conv(fshape=(5, 5, 32), init=init_uni, activation=Rectlin()),
          Pooling(fshape=2, strides=2),
          Conv(fshape=(5, 5, 32), init=init_uni, activation=Rectlin()),
          Pooling(fshape=2, strides=2),
          Dropout(keep=0.5),
          Affine(nout=256, init=init_uni, activation=Rectlin()),
          Dropout(keep=0.5),
          Affine(nout=10, init=init_uni, activation=Softmax())]
model = Model(layers)
cost = GeneralizedCost(costfunc=CrossEntropyMulti())
optimizer = GradientDescentMomentum(learning_rate=0.01, momentum_coef=0.9)
callbacks = Callbacks(model, eval_set=test_set, **args.callback_args)
model.fit(dataset=train_set, cost=cost, optimizer=optimizer, num_epochs=10, callbacks=callbacks)
error_pct = 100 * model.eval(test_set, metric=Misclassification())
print 'Misclassification error = %.1f%%' % error_pct
