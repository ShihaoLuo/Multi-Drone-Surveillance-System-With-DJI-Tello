from pic_match import PicMatch
import cv2 as cv
from pose_estimater import Demon
import matplotlib.pyplot as plt
import numpy as np

a = PicMatch()
img = cv.imread('./test/images/test0.jpg')

a.loaddata('./dataset/')
#a.showdataset()
obj, ratio = a.pic_match(img)
print(obj, ratio)
d = Demon()
img_ref = cv.imread('./dataset/'+obj+'/images/'+obj+'.jpg')
rot, vec = np.array(d.estimatepose(img, img_ref))
rot_inv = np.linalg.inv(rot)
vec = np.dot(rot, vec.reshape(-1,1))
print(rot, vec)
#d.showcloud()
