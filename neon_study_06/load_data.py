# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 10:57:39 2016

@author: Administrator
"""
import numpy as np
from PIL import Image
import os
import json


def load_data(traindir, points1_file, points2_file, imwidth):
    data = np.empty((4544, 3 * 256 * 256), dtype="int8")
    coordinate = np.empty((4544, 4), dtype="float32")  # [x1,y1,x2,y2]

    for idx in range(2):
        points_file = [points1_file, points2_file][idx]
        assert os.path.exists(points_file)
        points = json.load(file(points_file))
        for index, point in enumerate(points):
            assert len(point['annotations']) == 1
            path = os.path.join(traindir, point['filename'])
            im = Image.open(path)
            width, height = im.size
            coordinate[index][2 * idx] = 1.0 * point['annotations'][0]['x'] / width
            coordinate[index][2 * idx + 1] = 1.0 * point['annotations'][0]['y'] / height

            if idx == 0:
                im = im.resize((imwidth, imwidth))
                arr = np.asarray(im).reshape(196608)
                # arr = np.asarray(im)
                data[index] = arr
                # data[index, :, :, :] = [arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]]

    return data, coordinate


#traindir = 'train'
#imwidth = 256
#data, coordinate = load_data(traindir, 'points1.json', 'points2.json', imwidth)
#np.save('data.npy', data)
#np.save('coordinate.npy', coordinate)
