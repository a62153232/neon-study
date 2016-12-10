#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 21:49:42 2016

@author: auto-114
"""
import json
import os
import numpy as np
from PIL import Image
from aeon import DataLoader

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
    
bonnetfile='points1.json'
blowholefile='points2.json'
bonnets,blowholes=load(bonnetfile,blowholefile)
#traindir='train'
#imwidth=256
#x1map, y1map, x2map, y2map = read_annotations(traindir, 'points1.json', 'points2.json', imwidth)
#    

