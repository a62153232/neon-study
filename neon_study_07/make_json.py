#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 20:46:08 2016

@author: auto-114
"""

import csv
import json
import numpy as np
import json
import os
import numpy as np
from PIL import Image


def make_csv():
    train_json = []

    f_read = file('train.csv', 'rb')
    train = csv.reader(f_read)
    n = 0
    for line in train:
        if n == 0:
            # train_json.append(['Image','Annotations'])
            n = 1
            continue
        line[1] = '/home/auto-114/PycharmProjects/neon_study_10/annotations/' + line[0][0:-4] + '.json'
        line[0] = '/home/auto-114/PycharmProjects/neon_study_10/train/' + line[0]
        train_json.append(line)
    f_read.close()

    f_write = file('train_json.csv', 'wb')
    train = csv.writer(f_write)
    train.writerows(train_json)
    f_write.close()


def load(bonnetfile, blowholefile):
    bonnets = json.load(file(bonnetfile))
    blowholes = json.load(file(blowholefile))
    return bonnets, blowholes


def read_annotations(traindir, points1_file, points2_file, imwidth):
    # Read annotations
    xmap = [{}, {}]
    ymap = [{}, {}]
    for idx in range(2):
        points_file = [points1_file, points2_file][idx]
        assert os.path.exists(points_file)
        points = json.load(file(points_file))
        for point in points:
            assert len(point['annotations']) == 1
            path = os.path.join(traindir, point['filename'])
            im = Image.open(path)
            width, height = im.size
            xmap[idx][path] = int(
                1.0 * point['annotations'][0]['x'] * imwidth / width)
            ymap[idx][path] = int(
                1.0 * point['annotations'][0]['y'] * imwidth / height)
    return xmap[0], ymap[0], xmap[1], ymap[1]


def make_json(traindir, points1_file, points2_file, imwidth):
    assert os.path.exists(points1_file)
    assert os.path.exists(points2_file)
    points1 = json.load(file(points1_file))
    points2 = json.load(file(points2_file))
    for index, point in enumerate(points1):
        assert len(point['annotations']) == 1
        annotations = {
            "object": [
                {
                    "bndbox": {
                        "xmax": 0,
                        "xmin": 0,
                        "ymax": 0,
                        "ymin": 0,
                    },
                    "difficult": False,
                    "name": "head",
                },
            ],
        }
        path = os.path.join(traindir, point['filename'])
        im = Image.open(path)
        width, height = im.size
        annotations['object'][0]['bndbox']['xmax'] = int(1.0 * point['annotations'][0]['x'] * imwidth / width)
        annotations['object'][0]['bndbox']['xmin'] = int(1.0 * points2[index]['annotations'][0]['x'] * imwidth / width)
        annotations['object'][0]['bndbox']['ymax'] = int(1.0 * point['annotations'][0]['y'] * imwidth / height)
        annotations['object'][0]['bndbox']['ymin'] = int(1.0 * points2[index]['annotations'][0]['y'] * imwidth / height)

        f = open('/home/auto-114/PycharmProjects/neon_study_10/annotations/' + point['filename'][0:-4] + '.json', 'w')
        # im = im.resize((256, 256))
        # im.save('/home/auto-114/PycharmProjects/neon_study_10/train/' + point['filename'])
        json.dump(annotations, f)
        f.close()


# bonnetfile='points1.json'
# blowholefile='points2.json'
# traindir='train_pre'
#
# make_json(traindir, bonnetfile, blowholefile, 256)
# bonnets,blowholes=load(bonnetfile,blowholefile)

make_csv()
