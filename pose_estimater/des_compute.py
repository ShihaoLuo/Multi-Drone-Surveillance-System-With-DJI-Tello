#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 11:04:13 2020

@author: jake
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import json
import multiprocessing
#import set_world_point

object_name = '0--1-0'

def save_2_jason(_file, arr):
    data = {}
    cnt = 0
    for i in arr:
        data['KeyPoint_%d' % cnt] = []
        data['KeyPoint_%d' % cnt].append({'x': i.pt[0]})
        data['KeyPoint_%d' % cnt].append({'y': i.pt[1]})
        data['KeyPoint_%d' % cnt].append({'size': i.size})
        cnt += 1
    with open(_file, 'w') as outfile:
        json.dump(data, outfile)


def save_2_npy(_file, arr):
    np.save(_file, arr)


def read_from_jason(_file):
    result = []
    with open(_file) as json_file:
        data = json.load(json_file)
        cnt = 0
        while(data.__contains__('KeyPoint_%d' % cnt)):
            pt = cv.KeyPoint(x=data['KeyPoint_%d' % cnt][0]['x'],
                             y=data['KeyPoint_%d' % cnt][1]['y'],
                             _size=data['KeyPoint_%d' % cnt][2]['size'])
            result.append(pt)
            cnt += 1
    return result


def read_from_npy(_file):
    return np.load(_file)


def get_ROI(_img):
    img = _img
    for i in range(4):
        roi = cv.selectROI('roi', img, True, False)
        x, y, w, h = roi
        img[y:y+h, x:x+w] = [0,0,0]
    cv.destroyAllWindows()
    return img


img_query = cv.imread('./dataset/'+object_name+'/images/'+object_name+'.jpg')
img_query = get_ROI(img_query)
cv.imwrite('./dataset/'+object_name+'/images/'+object_name+'.jpg', img_query)
sift_paras = dict(nfeatures=0,
                 nOctaveLayers=3,
                 contrastThreshold=0.05,
                 edgeThreshold=10,
                 sigma=0.8)
'''surf_paras = dict(hessianThreshold=100,
                  nOctaves=10,
                  nOctaveLayers=2,
                  extended=1,
                  upright=0)
surf = cv.xfeatures2d.SURF_create(**surf_paras)'''
sift = cv.xfeatures2d.SIFT_create(**sift_paras)
kp_query, des_query = sift.detectAndCompute(img_query, None)
save_2_jason('dataset/'+object_name+'-kp.json',kp_query)
save_2_npy('dataset/'+object_name+'-des.npy',des_query)
print(len(kp_query))

