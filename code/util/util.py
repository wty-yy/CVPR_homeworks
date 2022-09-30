# -*- coding: utf-8 -*-

"""
@File    : util.py
@Author  : wtyyy
@Time    : 2022/9/26 8:33
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def img_open(fname):
    img = np.array(Image.open(fname))
    img = img[0:img.shape[0], 0:img.shape[1], :3]
    img_gray = np.expand_dims(np.array(Image.fromarray(img).convert('L')), -1)
    img = img.astype(np.float64) / 256  # 统一为[0,1)的数据范围
    img_gray = img_gray.astype(np.float64) / 256
    return img, img_gray


pi = np.pi
def gauss(x, y, sigma, dim=2):
    sigma_2 = sigma * sigma
    if dim == 2:
        return 1 / (2 * pi * sigma_2) * np.exp(-(x*x + y*y) / (2 * sigma_2))
    return 1 / (np.sqrt(2 * pi) * sigma) * np.exp(-(x*x) / (2 * sigma_2))

def gauss_filter(k, dim=2):  # 返回一个2k+1*2k+1的sigma=k的高斯滤波器
    n = 2 * k + 1
    filter = np.zeros([n, n, 1])
    for i in range(n):
        for j in range(n):
            if dim == 1 and i == k:
                filter[i, j, 0] = gauss(j-k, 0, sigma=k/2, dim=1)
            elif dim == 2:
                filter[i, j, 0] = gauss(i-k, j-k, k/2)
    return filter / np.sum(filter)

def gauss_filter_fixed(n, m, sigma):
    mid_n, mid_m = n // 2, m // 2
    filter = np.zeros([n, m, 1])
    for i in range(n):
        for j in range(m):
            filter[i, j, 0] = gauss(i-mid_n, j-mid_m, sigma)
    return filter / np.sum(filter)

def padding(img, dx, dy, mode=0):  # 原始图像img，横向增加dx，竖向增加dy个像素
    n, m, o = img.shape
    lx = (dx + 1) // 2  # 左侧填充量(向上取整)
    uy = (dy + 1) // 2  # 上侧填充量(向上取整)
    new_img = np.zeros([n + dx, m + dy, o])
    for i in range(n + dx):
        for j in range(m + dy):
            id1 = 0 if i < lx else (2 if i >= n + lx else 1)
            id2 = 0 if j < uy else (2 if j >= m + uy else 1)
            area = id2 * 3 + id1
            if mode == 0:
                if area != 4:
                    new_img[i, j, :] = np.zeros(o)
                else:
                    new_img[i, j, :] = img[i-lx, j-uy, :]
    return new_img

def conv(img, filter, mode=0):
    if img.ndim == 2:
        img = np.expand_dims(img, -1)
    n, m, o = img.shape
    a, b = filter.shape[0:2]
    output = np.zeros_like(img)
    if mode == 0:
        img = padding(img, a-1, b-1)
    for i in range(n):
        for j in range(m):
            for k in range(o):
                output[i, j, k] = np.sum(img[i:i+a, j:j+b, k] * filter[:,:,0])
    return output

def conv1(img, filter, mode=0):
    n, m, o = img.shape
    a, b = filter.shape[0:2]
    filter = filter.tolist()
    output = np.zeros_like(img).astype(np.float64).tolist()
    if mode == 0:
        img = padding(img, a-1, b-1)
    img = img.tolist()
    for i in range(n):
        for j in range(m):
            for k in range(o):
                for u in range(a):
                    for v in range(b):
                        output[i][j][k] += img[i+u][j+v][k] * filter[u][v][0]
    return np.array(output)

