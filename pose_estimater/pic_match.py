import numpy as np
import cv2 as cv
import json
import os
import matplotlib.pyplot as plt
import multiprocessing
import math

class PicMatch():
    def __init__(self, _algorithm='SIFT', min_match=30):
        self.algorithm = _algorithm
        self.kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        self.dataset = {}
        if _algorithm == 'SURF':
            self.detecter = cv.xfeatures2d.SURF_create(hessianThreshold=100, nOctaves=10, nOctaveLayers=2, extended=1, upright=0)
        elif _algorithm == 'SIFT':
            self.detecter = cv.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.05, edgeThreshold=10, sigma=0.8)


    def loaddata(self, _dataset_path):
        listdir = os.listdir(_dataset_path)
        for dir in listdir:
            if os.path.isdir(_dataset_path+dir):
                des = self.read_from_npy(_dataset_path+dir+'/'+dir+'-des.npy')
                kp = self.read_from_jason(_dataset_path+dir+'/'+dir+'-kp.json')
                _dataset = dict()
                _dataset['des'] = des
                _dataset['kp'] = kp
                self.dataset[dir] = _dataset

    def read_from_jason(self, _file):
        result = []
        with open(_file) as json_file:
            data = json.load(json_file)
            cnt = 0
            while (data.__contains__('KeyPoint_%d' % cnt)):
                pt = cv.KeyPoint(x=data['KeyPoint_%d' % cnt][0]['x'],
                                 y=data['KeyPoint_%d' % cnt][1]['y'],
                                 _size=data['KeyPoint_%d' % cnt][2]['size'])
                result.append(pt)
                cnt += 1
        return result

    def save_2_jason(self, _file, arr):
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

    def save_2_npy(self, _file, arr):
        np.save(_file, arr)

    def read_from_jason(self, _file):
        result = []
        with open(_file) as json_file:
            data = json.load(json_file)
            cnt = 0
            while (data.__contains__('KeyPoint_%d' % cnt)):
                pt = cv.KeyPoint(x=data['KeyPoint_%d' % cnt][0]['x'],
                                 y=data['KeyPoint_%d' % cnt][1]['y'],
                                 _size=data['KeyPoint_%d' % cnt][2]['size'])
                result.append(pt)
                cnt += 1
        return result

    def read_from_npy(self, _file):
        return np.load(_file)

    def pic_match(self, _img):
        img_test = _img
        kp_test, des_test = self.detecter.detectAndCompute(img_test, None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        ratio = 0
        match = ''
        for obj in self.dataset.keys():
            des_query = self.dataset[obj]['des']
            matches = flann.knnMatch(des_query, des_test, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
                    match_ratio = len(good)/len(des_query)
                    if match_ratio > ratio:
                        match = obj
                        ratio = match_ratio
        return match, ratio

    def show_pic(self, _img):
        fig = plt.figure(figsize=(12, 10))
        plt.subplot(1, 1, 1).axis("off")
        plt.imshow(_img)
        plt.show()

    def showdataset(self, _obj='all'):
        obj = _obj
        if _obj=='all':
            for i in self.dataset.keys():
                print('-----------dataset-----------\n')
                print('----------{}----------\n'.format(i))
                print(self.dataset[i])
        else:
            print('-----------dataset-----------\n')
            print('----------{}----------\n'.format(_obj))
            print(self.dataset[_obj])
