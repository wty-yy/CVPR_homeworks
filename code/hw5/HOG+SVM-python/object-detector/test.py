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
import numpy as np

from draw_figure import *
from skimage import feature, transform
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

def test_img():
    im = cv2.imread(r'mini_fox.jpg')
    downscale=1.5
    def sliding_window(im, window_size, step_size):
        for x in range(0, im.shape[0], step_size[0]):
            for y in range(0, im.shape[1], step_size[1]):
                yield x, y, im[x:x+window_size[0], y:y+window_size[1]]

    # Gauss金字塔
    for i, im_scaled in enumerate(transform.pyramid_gaussian(im, downscale=downscale, channel_axis=-1)):
        # 滑动窗口
        for x, y, im_window in sliding_window(im_scaled, (30, 100), (30, 10)):
            if im_window.shape[0] != 30 or im_window.shape[1] != 100:
                continue
            clone = im_scaled.copy()  # 在原图上重新绘制
            cv2.rectangle(clone, (y, x), (y + 100, x + 30), (255,255,255), thickness=2)  # 绘制窗口
            cv2.imshow(f"Sliding Window {im_scaled.shape}", clone)  # 显示窗口
            cv2.waitKey(20)  # 控制每帧长度
    cv2.waitKey()

def save_txt():
    a = np.arange(6).reshape(-1, 3)
    output = [('test1', '123', '321'), ('test2', '456', 654), *a.tolist()]
    np.savetxt('test.txt', output, fmt='%s')

if __name__ == '__main__':
    test_img()

