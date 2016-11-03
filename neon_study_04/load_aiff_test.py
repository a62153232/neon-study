# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 14:53:00 2016

@author: Administrator
"""

import aifc
import numpy as np
import scipy
import matplotlib


def ReadAIFF(file):
    s = aifc.open(file, 'r')
    nFrames = s.getnframes()
    strSig = s.readframes(nFrames)
    s.close()
    return np.fromstring(strSig, np.short).byteswap()


def ReadAIFF1(file):
    s = aifc.open(file, 'r')
    nFrames = s.getnframes()
    strSig = s.readframes(nFrames)
    s.close()
    wave_data = np.fromstring(strSig, dtype=np.short).byteswap()
    wave_data = 1. / nFrames * np.abs(scipy.fft(wave_data))
    return wave_data


data = ReadAIFF("kaggle_right_whale_calls\\whale_data\\data\\train\\train28111.aiff")
wave_data = ReadAIFF1("kaggle_right_whale_calls\\whale_data\\data\\train\\train28111.aiff")

plot1 = matplotlib.pyplot.plot(data)
matplotlib.pyplot.show(plot1)

plot2 = matplotlib.pyplot.plot(wave_data)  # 需要考虑如何归一化
matplotlib.pyplot.show(plot2)
#
#
#  f = aifc.open(filename, "r")
#  strsig = f.readframes(nframes)
#  f.close()
#  x = numpy.fromstring(strsig, numpy.short).byteswap()
#  y = 1. / nframes * numpy.abs(scipy.fft(x))
#  trainX[i, :] = y
