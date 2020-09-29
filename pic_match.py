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


MIN_MATH_COUNT = 25

img_test = cv.imread('./test_pic/0_1.7_250_0.jpg', 0)
img_query_1 = cv.imread('./env_pics/0_3_270_1.jpg', 0)

sift_para = dict(nfeatures=0,
                 nOctaveLayers=3,
                 contrastThreshold=0.05,
                 edgeThreshold=5,
                 sigma=1.6)

sift = cv.xfeatures2d.SIFT_create(**sift_para)
# kp_query, des_query = sift.detectAndCompute(img_query_1, None)
# save_2_jason('data.jason', kp_query)
# save_2_npy('data_des.npy', des_query)
kp_test, des_test = sift.detectAndCompute(img_test, None)
kp_query_1 = read_from_jason('kp_goodm.jason')
des_query_1 = read_from_npy('des_goodm.npy')
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des_query_1, des_test, k=2)
print('the num of finding featurs of query is {}\n'.format(len(des_query_1)))
print('the num of finding featurs of test is {}\n'.format(len(des_test)))
print('the num of finding matches is {}\n'.format(len(matches)))

good = []
kp_good_match_query = []
des_good_match_query = []
for m, n in matches:
    if m.distance < 0.5*n.distance:
        good.append(m)
        print('--------------------\n')
        print('m.imgIdx: {}\n'.format(m.imgIdx))
        print('m.queryIdx: {}\n'.format(m.queryIdx))
        print('m.trainIdx: {}\n'.format(m.trainIdx))
        print('kp_query: {}\n'.format(kp_query_1[m.queryIdx].pt))
        print('kp_test: {}\n'.format(kp_test[m.trainIdx].pt))
        kp_good_match_query.append(kp_query_1[m.queryIdx])
        des_good_match_query.append(des_query_1[m.queryIdx])
print("the len of good match is {}\n".format(len(good)))
# save_2_jason('kp_goodm.jason',kp_good_match_query)
# save_2_npy('des_goodm.npy',des_good_match_query)

if len(good)>MIN_MATH_COUNT:
    src_pts = np.float32([kp_good_match_query[i].pt for i in range(len(kp_good_match_query))]).reshape(-1,1,2)
    #src_pts = np.float32([kp_query_1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp_test[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img_query_1.shape
    d = 1
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img_test = cv.polylines(img_test,[np.int32(dst)],True,255,3,cv.LINE_AA)

else:
    print("Not enough matchs are found - {}/{}".format(len(good),MIN_MATH_COUNT))
    matchesMask = None
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = None,
                   matchesMask = matchesMask,
                   flags = 2)
img = cv.drawMatches(img_query_1,kp_query_1,img_test,kp_test,good,None,**draw_params)

cv.imshow(' ',img)

img = cv.drawKeypoints(img_query_1, kp_query_1, None)
plt.imshow(img)

rvec = np.array([[0],
         [3.14],
         [0]])
tvec = np.array([[0.0],
         [0.0],
         [300.0]])

point_world = np.array([[62.5, 85, 63],
                        [8.5, 40, 40],
                        [18.5, 79, 41.5],
                        [-83, 47, 0],
                        [0, 168.5, 0]])
point_pixel = np.array([[315.800537109375, 349.0028381347656],
                        [446.271240234375, 259.53466796875],
                        [431.7677917480469, 317.7646179199219],
                        [649.1941528320312, 231.0972137451172],
                        [477.6302490234375, 481.7046203613281]])
camera_matrix = np.load('camera_matrix_tello.npy')
distor_matrix = np.load('distor_matrix_tello.npy')
pnppara = dict(objectPoints=point_world,
               imagePoints=point_pixel,
               cameraMatrix=camera_matrix,
               distCoeffs=distor_matrix,
               #rvec=rvec,
               #tvec=tvec,
               #useExtrinsicGuess=True,
               flags=cv.SOLVEPNP_ITERATIVE)
_, rvec, tvec = cv.solvePnP(**pnppara)
rotM = np.matrix(cv.Rodrigues(rvec)[0])
pose = -rotM.I*np.matrix(tvec)
print(pose)
print('----------\n')
