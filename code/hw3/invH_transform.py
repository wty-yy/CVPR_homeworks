# -*- coding: utf-8 -*-

"""
@File    : invH_transform.py
@Author  : wtyyy
@Time    : 2022/10/30 22:47
"""

import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from PIL import Image
from util.util import *
from util.draw_figure import draw_some
from util.resizing import *
import numpy as np
from tqdm import tqdm

def transform(img, H):
    H = H.T
    n, m, c = img.shape
    mid = np.array([n/2, m/2, 0])
    H[2, :] += mid @ H - mid
    output = np.zeros(img.shape)
    for i in tqdm(range(n)):
        for j in range(m):
            x, y, z = (np.array([j, i, 1]) @ H)
            x /= z  # 正则化
            y /= z
            output[i, j, :] = bilinear_interpolate(img, y, x)  # 利用双线性插值
    return output

def open_img(fname):
    img = np.array(Image.open(fname))
    img = np.expand_dims(img, -1)
    return img
img1 = open_img('Image1.tif')
img3 = open_img('Image3.tif')
#t2 = Transform(img2)
H1 = np.array([[-0.2204, 0.8074, 166.6303],
               [0.2787, 0.0759, 178.1498],
               [-0.0009, 0.0001, 1.0000]])
H3 = np.array([[-0.3254, 0.8275, 197.7143],
               [0.5622, 0.1381, 95.3857],
               [-0.0006, -0.0000, 1.0000]])
draw_some((img1, 'Image1', 'line'), (transform(img1, H1), '逆变换', 'line'))
# draw_some((img3, 'Image3', 'line'), (transform(img3, H3), '逆变换', 'line'))

