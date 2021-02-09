# -*- coding: utf-8 -*-
# @Time    : 2021/2/9 下午5:20
# @Author  : JakeShihao Luo
# @Email   : jakeshihaoluo@gmail.com
# @File    : face_test.py
# @Software: PyCharm
import face_recognition
import cv2 as cv
import matplotlib.pyplot as plt

# image = cv.imread('./249.0-284.0-228.0-359.0.jpg')
# image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
# cv.imwrite('./11111.jpg', image)
image = face_recognition.load_image_file('./504.0-644.0-236.0-1.0.jpg')
face_locations = face_recognition.face_locations(image)
print(face_locations)
plt.imshow(image)
plt.show()
