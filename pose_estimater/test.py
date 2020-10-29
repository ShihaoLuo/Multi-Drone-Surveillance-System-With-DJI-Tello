from pic_match import PicMatch
import cv2 as cv
from pose_estimater import Demon
import matplotlib.pyplot as plt
import numpy as np
import math

ROTTOW = dict()
ROTTOW['1-0-0'] = np.array([[1, 0, 0],
                            [0, 0, 1],
                            0, -1, 0])
ROTTOW['0-1-0'] = np.array([[0, 0, -1],
                         [1, 0, 0],
                            [0, -1, 0]])
ROTTOW['-1-0-0'] = np.array([[-1, 0, 0],
                             [0, 0, -1],
                             [0, -1, 0]])
ROTTOW['0--1-0'] = np.array([[0, 0, 1],
                             [-1, 0, 0],
                             [0, -1, 0]])



a = PicMatch()
img = cv.imread('./test/images/test0.jpg')
#obj = '-90-0--300'
a.loaddata('./dataset/')
#a.showdataset()
obj, ratio = a.pic_match(img)
print(obj, ratio)
d = Demon()
img_ref = cv.imread('./dataset/'+obj+'/images/'+obj+'.jpg')
rot, vec = np.array(d.estimatepose(img, img_ref), dtype=object)
print(rot, vec)
rot = np.dot(ROTTOW[obj], rot)
#rot_inv = np.linalg.inv(rot)
vec = np.dot(rot, vec.reshape(-1,1))*2
angle = math.atan2(rot[1, 2], rot[0, 2])*180.0/math.pi
print(angle, vec)
d.showcloud()

