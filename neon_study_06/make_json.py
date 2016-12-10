#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 20:46:08 2016

@author: auto-114
"""

import csv
import numpy as np

train_json=[]

f_read=file('train.csv','rb')
train=csv.reader(f_read)
n=0
for line in train:
    if n==0:
        train_json.append(['Image','Annotations'])
        n=1
        continue    
    line[1]='/annotations/'+line[0][0:-4]+'.json'
    line[0]='/train/'+line[0]
    train_json.append(line)
f_read.close()
 
f_write=file('train_json.csv','wb')
train=csv.writer(f_write)
train.writerows(train_json)
f_write.close()