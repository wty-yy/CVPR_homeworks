# -*- coding: utf-8 -*-

"""
@File    : transform.py
@Author  : wtyyy
@Time    : 2022/10/4 18:00
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.split(sys.path[0])[0])
from util.util import *
from util.draw_figure import draw_some


class Transform:
    def __init__(self, img):
        self.img = img
        self.n, self.m, self.c = img.shape
    def translation(self, t1, t2):  # 平移变换
        return np.array([[1, 0, t1], [0, 1, t2], [0, 0, 1]]).astype('float64')

    def rotation(self, theta):  # 旋转变换
        return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]).astype('float64')

    def euclidean(self, theta, t1, t2):  # 欧式变换，旋转+平移
        ret = self.rotation(theta)
        ret[:, 2] = np.array([t1, t2, 1]).flatten().astype('float64')
        return ret

    def similitude(self, s, theta, t1, t2):  # 相似变换，缩放+旋转+平移
        ret = self.rotation(theta) * s
        ret[:, 2] = np.array([t1, t2, 1]).flatten().astype('float64')
        return ret

    def affine(self, a11, a12, a21, a22, t1, t2):  # 仿射变换
        return np.array([[a11, a12, t1], [a21, a22, t2], [0, 0, 1]]).astype('float64')

    def bilinear_interpolate(self, x, y):
        def calc_area(x0, y0):
            return abs(x - x0) * abs(y - y0)
        if x < 0 or x > self.n-1 or y < 0 or y > self.m-1:
            return 0
        x0, y0 = int(x), int(y)
        x0 = x0 - 1 if x0 == self.n-1 else x0
        y0 = y0 - 1 if y0 == self.m-1 else y0
        pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
        ret = np.zeros(3)
        for i in range(4):
            ret += self.img[x0+pos[3-i][0], y0+pos[3-i][1], :] * calc_area(x0 + pos[i][0], y0 + pos[i][1])
        return ret

    def transform(self, T, tomid=False):
        if tomid:
            mid = np.array([self.n/2, self.m/2, 1]).reshape([-1, 1])
            T[:, 2] += (mid - T @ mid).flatten()
        invT = np.linalg.inv(T)
        output = np.zeros_like(img)
        for i in tqdm(range(self.n)):
            for j in range(self.m):
                x, y = (invT @ np.array([i, j, 1]).reshape([-1, 1]))[0:2, 0]
                output[i, j, :] = self.bilinear_interpolate(x, y)
        return output

img, img_gray = img_open('../figure/fox1.png')
n, m, c = img.shape
# draw_some((img, '原图'), (padding(img, 200, 200, mode=0), '零填充'),
#           (padding(img, 200, 200, mode=1), '边界环绕'),
#           (padding(img, 200, 200, mode=2), '边界复制'),
#           (padding(img, 200, 200, mode=3), '边界镜像'))
# 几何变换
# t = Transform(img)
# draw_some((img, '原图'), (t.transform(t.translation(n/4, m/4)), '平移(N/4,M/4)'),
#           (t.transform(t.rotation(np.pi/6)), '旋转$\pi/6$'),
#           (t.transform(t.euclidean(np.pi/4, 0, 0), tomid=True), '旋转$\pi/4$'),
#           (t.transform(t.similitude(0.5, np.pi/4, 0, 0), tomid=True), '旋转$\pi/4$'),
#           (t.transform(t.similitude(2, np.pi/4, 0, 0), tomid=True), '缩放2，旋转$\pi/4$'),
#           (t.transform(t.affine(0.5, 1, 1, 0.5, 0, 0), tomid=True), '仿射a=[0.5,1,1,0.5]'),
#           (t.transform(t.affine(-0.5, 1, -1, 0.5, 0, 0), tomid=True), '仿射a=[-0.5,1,-1,0.5]'),
#           (t.transform(t.affine(0.5, 1, 1, -0.5, 0, 0), tomid=True), '仿射a=[0.5,1,1,-0.5]'), shape=(3, 3))
# Gauss金字塔
filter = gauss_filter(2)
G1 = conv(img, filter, stride=2)
G2 = conv(G1, filter, stride=2)
G3 = conv(G2, filter, stride=2)
draw_some((img, '原图$G_0$'), (G1, '$G_1$'), (G2, '$G_2$'), (G3, '$G_3$'), origin=True, shape=(1, 2))