# -*- coding: UTF-8 -*-

"""
@author: wty-yy
@software: PyCharm
@file: test.py.py
@time: 2022/12/2 20:12
"""

import json
import configparser as cp
import sys
from pathlib import Path
import cv2
import joblib
import matplotlib.pyplot as plt
from draw_figure import *
from skimage import feature
from config import *

def feat():
    path = Path(r'../data/features/pos')
    for feat_path in path.glob('*.feat'):
        # print(feat_path)
        fd = joblib.load(feat_path)
        print(fd.shape)
        break

def hog_show():
    path = Path(r'../data/dataset/PNGImages/pos/pos-0.png')
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    fd, hog_img = feature.hog(img, orientations=orientations,
                     pixels_per_cell=pixels_per_cell,
                     cells_per_block=cells_per_block,
                     visualize=True)
    #plt.imshow(hog_img, cmap='gray')
    fig = plt.figure(figsize=(12,6))
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('图像1')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 3)
    plt.imshow(hog_img, cmap='gray')
    plt.title('Hog1')
    plt.xticks([])
    plt.yticks([])

    path = Path(r'../data/dataset/PNGImages/pos/pos-5.png')
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    fd, hog_img = feature.hog(img, orientations=orientations,
                              pixels_per_cell=pixels_per_cell,
                              cells_per_block=cells_per_block,
                              visualize=True)
    #plt.imshow(hog_img, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(img, cmap='gray')
    plt.title('图像2')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 4)
    plt.imshow(hog_img, cmap='gray')
    plt.title('Hog2')
    plt.xticks([])
    plt.yticks([])
    fig.tight_layout()
    plt.savefig('hog特征提取.png', dpi=300)
    plt.show()

    print('特征向量维度', fd.shape)

if __name__ == '__main__':
    hog_show()
