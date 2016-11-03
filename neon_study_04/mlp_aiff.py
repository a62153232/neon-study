import aifc
import numpy as np
import pandas
from neon.callbacks.callbacks import Callbacks
from neon.initializers import Gaussian
from neon.layers import GeneralizedCost, Affine, DeepBiRNN
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, CrossEntropyBinary, Misclassification, Logistic
from neon.util.argparser import NeonArgparser
from neon import logger as neon_logger
from neon.data import ArrayIterator, DataLoader


def ReadAIFF(file):
    wave = aifc.open(file, 'r')
    nFrames = wave.getnframes()
    wave_str = wave.readframes(nFrames)
    wave.close()
    wave_data = np.fromstring(wave_str, dtype=np.short).byteswap()
    wave_data = wave_data.astype(np.float32)
    w_max = max(wave_data)
    w_min = min(wave_data)
    # wave_data = wave_data.astype(np.float32)
    wave_data = (wave_data.astype(np.float32) - w_min) / (w_max - w_min)
    #    wave_data = 1. / nFrames * np.abs(scipy.fft(wave_data))
    # wave_data_gram = pylab.specgram(wave_data)
    return wave_data


def ReadCSV(file):
    f = pandas.read_csv(file)
    train_y = f["label"]
    return train_y


def shuffle(X, y):
    data_column_stack = np.column_stack((X, y))
    np.random.shuffle(data_column_stack)
    X = data_column_stack[:, :-1]
    y = data_column_stack[:, -1]
    return X, y


def load_data(path):
    train_dim = 1000
    test_dim = 500
    nframes = 4000
    train_X = np.zeros((train_dim, nframes), dtype=np.float32)
    for i in range(0, train_dim):
        filename = path + "train/train%d.aiff" % (i + 1)
        wave_data = ReadAIFF(filename)
        # wave_data_gram = wave_data_gram.reshape(nframes)
        # wave_data_gram_max = max(wave_data_gram)
        train_X[i, :] = wave_data

        train_y = ReadCSV("data/train.csv")
        train_y = np.array(train_y[0:1000])

        # test_X = np.zeros((test_dim, nframes), dtype=np.float32)
        #    for i in range(0, test_dim):
        #        filename = path+"test/test%d.aiff" % (i+1)
        #        wave_data_gram = ReadAIFF(filename)
        #        test_X[i, :] = wave_data_gram.reshape(nframes)

    train_X, train_y = shuffle(train_X, train_y)
    return train_X[0:700], train_y[0:700], train_X[700:1000], train_y[700:1000]


parser = NeonArgparser(__doc__)  # Use gpu
args = parser.parse_args()
args.epochs = 50

train_set_x, train_set_y, valid_set_x, valid_set_y = load_data("data/")
train_set = ArrayIterator(train_set_x, train_set_y, nclass=2)
valid_set = ArrayIterator(valid_set_x, valid_set_y, nclass=2)

init_uni = Gaussian(loc=0.0, scale=0.01)

# setup model layers
layers = [DeepBiRNN(4000, init=init_uni, activation=Rectlin()),
          DeepBiRNN(1000, init=init_uni, activation=Rectlin()),
          DeepBiRNN(200, init=init_uni, activation=Rectlin()),
          Affine(nout=2, init=init_uni, activation=Logistic(shortcut=True))]

# setup cost function as CrossEntropy
cost = GeneralizedCost(costfunc=CrossEntropyBinary())

# setup optimizer
optimizer = GradientDescentMomentum(0.1, momentum_coef=0.9, stochastic_round=args.rounding)

# initialize model object
mlp = Model(layers=layers)

# configure callbacks
callbacks = Callbacks(mlp, eval_set=valid_set, **args.callback_args)

# run fit
mlp.fit(train_set, optimizer=optimizer,
        num_epochs=args.epochs, cost=cost, callbacks=callbacks)
error_rate = mlp.eval(valid_set, metric=Misclassification())
neon_logger.display('Misclassification error = %.1f%%' % (error_rate * 100))
