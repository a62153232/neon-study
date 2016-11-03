from neon.callbacks.callbacks import Callbacks
from neon.initializers import Gaussian, Uniform
from neon.layers import GeneralizedCost, Affine, Conv, Pooling, Dropout, DeepBiLSTM
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, CrossEntropyBinary, Misclassification, Softmax, Logistic
from neon.util.argparser import NeonArgparser
from neon import logger as neon_logger
from neon.data import ArrayIterator
from neon.backends import gen_backend
from load_aiff import load_data

parser = NeonArgparser(__doc__)  # Use gpu
args = parser.parse_args(gen_be=False)
args.epochs = 13
args.batch_size = 200

gen_backend(backend=args.backend,  # Modify backend
            rng_seed=args.rng_seed,
            device_id=args.device_id,
            batch_size=args.batch_size,
            datatype=args.datatype,
            max_devices=args.max_devices,
            compat_mode=args.compat_mode)

train_set_x, train_set_y, valid_set_x, valid_set_y = load_data("data/")
train_set = ArrayIterator(train_set_x, train_set_y, nclass=2, lshape=(1, 65, 122))
valid_set = ArrayIterator(valid_set_x, valid_set_y, nclass=2, lshape=(1, 65, 122))
# init_uni = Gaussian(loc=0.0, scale=0.01)
init_uni = Uniform(low=-0.1, high=0.1)

# setup model layers
hidden_size = 128
layers1 = [DeepBiLSTM(hidden_size, init_uni, activation=Rectlin(), gate_activation=Logistic()),
           DeepBiLSTM(hidden_size, init_uni, activation=Rectlin(), gate_activation=Logistic()),
           DeepBiLSTM(hidden_size, init_uni, activation=Rectlin(), gate_activation=Logistic()),
           Affine(2, init=init_uni, activation=Softmax())]

layers2 = [Conv(fshape=(7, 7, 32), init=init_uni, activation=Rectlin()),
           Pooling(fshape=2, strides=2),
           Conv(fshape=(5, 5, 32), init=init_uni, activation=Rectlin()),
           Pooling(fshape=2, strides=2),
           Conv(fshape=(3, 3, 32), init=init_uni, activation=Rectlin()),
           Affine(nout=200, init=init_uni, activation=Rectlin()),
           Affine(nout=2, init=init_uni, activation=Logistic())]

# setup cost function as CrossEntropy
cost = GeneralizedCost(costfunc=CrossEntropyBinary())

# setup optimizer
optimizer = GradientDescentMomentum(
    0.1, momentum_coef=0.9, stochastic_round=args.rounding)

# initialize model object
mlp = Model(layers=layers2)

# configure callbacks
callbacks = Callbacks(mlp, eval_set=valid_set, **args.callback_args)

# run fit
mlp.fit(train_set, optimizer=optimizer,
        num_epochs=args.epochs, cost=cost, callbacks=callbacks)
error_rate = mlp.eval(valid_set, metric=Misclassification())
neon_logger.display('Misclassification error = %.1f%%' % (error_rate * 100))
