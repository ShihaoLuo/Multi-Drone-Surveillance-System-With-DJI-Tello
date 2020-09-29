#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 13:22:43 2020

@author: jake
"""

import numpy as np
import cv2 as cv
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing


class Marker_Manager:
    def __init__(self):
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_7X7_50)
        self.camera_matrix = np.load('camera_matrix_tello.npy')
        self.distor_matrix = np.load('distor_matrix_tello.npy')
        self.params = aruco.DetectorParameters_create()
        self.markerId = None
        self.markerCorners = None
        self.rejectedCandidates = None
        self.frame_marker = multiprocessing.Queue()
        self.show_thread = multiprocessing.Process(target=self.show_pic_thread)
        self.show_thread.start()

    def generate_marker(self, id, _file):
        markerImage = np.zeros((800, 800), dtype=np.uint8)
        markerImage = cv.aruco.drawMarker(self.aruco_dict, id, 800, markerImage, 1)
        fig = plt.figure(figsize=(11.7, 16.5))
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(markerImage, cmap=mpl.cm.gray, interpolation="nearest")
        ax.axis("off")
        plt.savefig(_file)
        return markerImage

    def marker_detect(self, frame):
        self.markerId = None
        self.markerCorners = None
        self.rejectedCandidates = None
        self.markerCorners, self.markerIds, self.rejectedCandidates = cv.aruco.detectMarkers(frame,
                                                                                             self.aruco_dict,
                                                                                             parameters=self.params)
        # print('markerId is :{}\nmarkerconers is :{}\n'.format(self.markerIds, self.markerCorners))

    def drawdetectedmarker(self, frame):
        self.markerId = None
        self.markerCorners = None
        self.rejectedCandidates = None
        self.marker_detect(frame)
        print('markerId is :{}\nmarkerconers is :{}\n'.format(self.markerIds, self.markerCorners))
        if self.markerIds is not None:
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(self.markerCorners,
                                                            0.094,
                                                            self.camera_matrix,
                                                            self.distor_matrix)
            (rvec-tvec).any()
            for i in range(rvec.shape[0]):
                aruco.drawAxis(frame,
                               self.camera_matrix,
                               self.distor_matrix,
                               rvec[i, :, :],
                               tvec[i, :, :],
                               0.03)
                aruco.drawDetectedMarkers(frame, self.markerCorners)
            self.frame_marker.put(frame)

    def show_pic_thread(self):
        while True:
            if not self.frame_marker.empty():
                cv.imshow("marker", self.frame_marker)
                key = cv.waitKey(20)
                if key == ord('q'):
                    cv.destroyWindow('marker')
                    break

    def estimate_pose(self, point_world):
        pnppara = dict(objectPoints=point_world,
                       imagePoints=self.markerCorners[0],
                       cameraMatrix=self.camera_matrix,
                       distCoeffs=self.distor_matrix,
                       # rvec=rvec,
                       # tvec=tvec,
                       # useExtrinsicGuess=True,
                       flags=cv.SOLVEPNP_ITERATIVE)
        _, rvec, tvec = cv.solvePnP(**pnppara)
        rotM = np.matrix(cv.Rodrigues(rvec)[0])
        pose = -rotM.I*np.matrix(tvec)
        return pose


'''a = Marker_Manager()
a.generate_marker(0, 'Marker_0.pdf')'''
