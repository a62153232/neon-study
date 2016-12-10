# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 21:37:39 2016

@author: Administrator
"""
import aifc
import numpy as np
import scipy
import pandas
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

NFFT = 400
noverlap = NFFT * (1 - 1. / 10)
noverlap = NFFT * 0.95
log_scale = 10 ** 0
shape = (201, 181)


def ReadAIFF(file):
    wave = aifc.open(file, 'r')
    nFrames = wave.getnframes()
    wave_str = wave.readframes(nFrames)
    wave.close()
    wave_data = np.fromstring(wave_str, dtype=np.short).byteswap()
    #    wave_data = 1. / nFrames * np.abs(scipy.fft(wave_data))
    wave_data_gram = pylab.specgram(wave_data, NFFT=NFFT, noverlap=noverlap)[0]
    # wave_data_gram = np.log(1 + log_scale * wave_data_gram)
    print wave_data_gram
    wave_data_gram = wave_data_gram.reshape(shape[0] * shape[1])
    w_max = max(wave_data_gram)
    w_min = min(wave_data_gram)
    wave_data_gram = (wave_data_gram - w_min) / (w_max - w_min)
    return wave_data_gram


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
    train_dim = 1500
    test_dim = 500
    nframes = shape[0] * shape[1]
    train_X = np.zeros((train_dim, nframes), dtype=np.float32)
    for i in range(0, train_dim):
        filename = path + "train/train%d.aiff" % (i + 1)
        wave_data_gram = ReadAIFF(filename)
        train_X[i, :] = wave_data_gram

        train_y = ReadCSV("data/train.csv")
        train_y = np.array(train_y[0:1500])

        # test_X = np.zeros((test_dim, nframes), dtype=np.float32)
        #    for i in range(0, test_dim):
        #        filename = path+"test/test%d.aiff" % (i+1)
        #        wave_data_gram = ReadAIFF(filename)
        #        test_X[i, :] = wave_data_gram.reshape(nframes)

    train_X, train_y = shuffle(train_X, train_y)

    return train_X[0:1000], train_y[0:1000], train_X[1000:1500], train_y[1000:1500]


# X = np.random.randint(10, 20, (5, 5))
# y = np.random.randint(0, 2, 5)
# print X
# print y
# X, y = shuffle(X, y)
# print X
# print y

from PIL import Image

data = ReadAIFF("data/train/train19397.aiff")
data_nocall = ReadAIFF("data/train/train10.aiff")
data = data + 0.28 * data_nocall
w_max = max(data)
w_min = min(data)
data = 255 * (data - w_min) / (w_max - w_min)
data = data.reshape(shape).astype(np.int8)
print data
im = Image.fromarray(data)
print im.format, im.mode
im.save("aiff_129x118.gif")
# fig = plt.figure()
# ax = fig.add_subplot(221)
# ax.imshow(data)
# plt.show()
