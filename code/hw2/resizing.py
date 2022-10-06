# -*- coding: utf-8 -*-

"""
@File    : resizing.py
@Author  : wtyyy
@Time    : 2022/10/4 18:00
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from queue import Queue

sys.path.append(os.path.split(sys.path[0])[0])
from util.util import *
from util.draw_figure import draw_some
from util.filter_fourier import *


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

    def transform(self, T, tomid=False):
        if tomid:
            mid = np.array([self.n/2, self.m/2, 1]).reshape([-1, 1])
            T[:, 2] += (mid - T @ mid).flatten()
        invT = np.linalg.inv(T)
        output = np.zeros_like(img)
        for i in tqdm(range(self.n)):
            for j in range(self.m):
                x, y = (invT @ np.array([i, j, 1]).reshape([-1, 1]))[0:2, 0]
                output[i, j, :] = bilinear_interpolate(self.img, x, y)
        return output


def bilinear_interpolate(img, x, y):  # 双线性插值
    n, m, c = img.shape
    def calc_area(x0, y0):
        return abs(x - x0) * abs(y - y0)
    if x < 0 or x > n - 1 or y < 0 or y > m - 1:
        return 0
    x0, y0 = int(x), int(y)
    x0 = x0 - 1 if x0 == n - 1 else x0
    y0 = y0 - 1 if y0 == m - 1 else y0
    pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    ret = np.zeros(c)
    for i in range(4):
        ret += img[x0 + pos[3 - i][0], y0 + pos[3 - i][1], :] * calc_area(x0 + pos[i][0], y0 + pos[i][1])
    return ret

def down_resolution(img, k):  # 使用高斯核大小为2k+1x2k+1，降低1/2的分辨率
    filter = gauss_filter(2)
    return conv(img, filter, stride=2)

def up_resolution(img, padding_mode=3):  # 使用高斯核大小为5x5，提高1/2的分辨率，默认使用边界镜像填充
    n, m, c = img.shape
    shape = np.array(img.shape)
    shape[0:2] *= 2
    out = np.zeros(shape)
    for i in range(n):
        for j in range(m):
            out[i*2, j*2, :] = img[i, j, :]
    return conv(out, gauss_filter(5), padding_mode=padding_mode) * 4

def sampling(img, stride):  # 对图像进行步长为stride的采样
    n, m, c = img.shape
    out = np.zeros([n//stride, m//stride, c])
    for i in range(0, n, stride):
        for j in range(0, m, stride):
            out[i//stride, j//stride, :] = img[i, j, :]
    return out

def img_gradient(img, sigma):  # 方差为sigma的一阶微分Gauss核，对图像做微分，返回x方向的梯度, y方向的梯度, 幅度图和相位图
    x_filter = gauss_filter(int(sigma*2), derivative=0)
    y_filter = gauss_filter(int(sigma*2), derivative=1)
    x_img = conv(img, x_filter, padding_mode=3)
    y_img = conv(img, y_filter, padding_mode=3)
    magnitude = np.sqrt(np.power(x_img, 2) + np.power(y_img, 2))
    orientation = np.arctan2(x_img, y_img)
    return x_img, y_img, magnitude, orientation

def NMS_derivative(img, sigma, show_info=True):
    n, m, c = img.shape
    _, _, magnitude, orientation = img_gradient(img, sigma)
    magnitude /= magnitude.mean()
    del_cnt = 0
    nms = magnitude.copy()  # 非极大值像素梯度抑制
    for i in tqdm(range(n)):
        for j in range(m):
            for k in range(c):  # 仅处理灰度图像，所以c=1
                p = (i + np.cos(orientation[i, j, k]), j + np.sin(orientation[i, j, k]))
                q = (i - np.cos(orientation[i, j, k]), j - np.sin(orientation[i, j, k]))
                d_p = bilinear_interpolate(magnitude, *p)
                d_q = bilinear_interpolate(magnitude, *q)
                if nms[i, j, k] <= max(int(d_p), int(d_q)):
                    nms[i, j, k] = 0
                    del_cnt += 1
    if show_info:
        print(f'NMS过程共计删除{del_cnt}个像素点，占比{del_cnt / (n * m):%}')
    return magnitude, orientation, nms

def canny(img, sigma, show_info=True):  # Canny边缘检测算法，sigma为Gauss核方差，show为是否显示计算过程
    n, m, c = img.shape
    magnitude, orientation, nms = NMS(img, sigma, show_info)
    high = np.percentile(magnitude, 95)  # 高阈值
    low = np.percentile(magnitude, 90)  # 低阈值
    high_img = magnitude * (magnitude >= high)
    low_img = magnitude * (magnitude >= low)
    ret = np.zeros_like(img)
    dx = [-1, -1, -1, 0, 0, 1, 1, 1]  # 8个方向
    dy = [-1, 0, 1, -1, 1, -1, 0, 1]
    total_connect = 0

    def bfs(x, y):  # 利用广搜进行边缘连接
        count = 0
        q = Queue()
        q.put((x, y))
        while not q.empty():
            x, y = q.get()
            if ret[x, y, 0] != 0:
                continue
            ret[x, y, 0] = 1
            count += 1
            for i in range(8):
                tx, ty = x + dx[i], y + dy[i]
                if tx < 0 or tx > n-1 or ty < 0 or ty > m-1:
                    continue
                if low_img[tx, ty] != 0:
                    q.put((tx, ty))
        return count

    for i in tqdm(range(n)):
        for j in range(m):
            for k in range(c):
                if high_img[i, j, k] != 0:
                    total_connect += bfs(i, j)
    if show_info:
        print(f'边缘连接过程共计连接{total_connect}个像素点，占比{total_connect/(n*m):%}')
        draw_some((magnitude, '幅度图', 'line'), (orientation, '方向图', 'clip', 'hot'), (nms, 'NMS', 'line'),
                  (high_img, '高阈值处理', 'line'), (low_img, '低阈值处理', 'line'), (ret, 'Canny边缘检测结果', 'line'), shape=(2, 3))
    return ret

def draw_dot(img, x, y):
    dx = [-1, -1, -1, 0, 0, 0, 1, 1, 1]  # 9个方向
    dy = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
    for _ in range(9):
        tx, ty = x + dx[_], y + dy[_]
        if tx < 0 or tx > n - 1 or ty < 0 or ty > m - 1:
            continue
        img[tx, ty, :] = np.array([0, 1, 0])

def harris(img, sigma, alpha, k=5, show_img=None):  # img:图像，sigma:Gauss核方差，alpha:响应函数系数，k:NMS大小2k+1x2k+1
    d_x, d_y, _, _ = img_gradient(img, 0.5)
    d_x *= 100
    d_y *= 100
    filter = gauss_filter(int(sigma * 2))
    A = conv(d_x * d_x, filter)
    B = conv(d_x * d_y, filter)
    C = conv(d_y * d_y, filter)
    R = np.abs(A * C - B * B - alpha * (A + C))
    corner = img * (R > np.percentile(R, 98))
    nms = np.zeros_like(corner)
    n, m, c = img.shape
    pad = padding(corner, 2*k, 2*k, mode=0)
    for i in range(n):
        for j in range(m):
            for o in range(c):
                # if corner[i, j, o] != 0:
                #     print(corner[i, j, o], np.max(pad[i:i+2*k, j:j+2*k, o]))
                if corner[i, j, o] == np.max(pad[i:i+2*k, j:j+2*k, o]) and corner[i, j, o] != 0:
                    nms[i, j, o] = 1
                    if show_img is not None:
                        if len(show_img.shape) == 1:
                            _, show_img = img_change(show_img)
                        draw_dot(show_img, i, j)
    draw_some((img, '原图'), (R, '响应函数', 'upper'), (nms, '98%阈值分离+NMS'), (show_img, '标记角点'))
    return show_img

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
# # Gauss金字塔
# G1 = down_resolution(img, 2)
# G2 = down_resolution(G1, 2)
# G3 = down_resolution(G2, 2)
# G4 = down_resolution(G3, 2)
# draw_some((img, '原图$G_0$'), (G1, '$G_1$'), (G2, '$G_2$'), (G3, '$G_3$'), (G4, '$G_4$'), origin=True, shape=(1, 2))
# # Laplace金字塔
# g3 = up_resolution(G4)
# g2 = up_resolution(G3)
# g1 = up_resolution(G2)
# g0 = up_resolution(G1)
# draw_some((g0, '$g0$'), (g1, '$g1$'), (g2, '$g2$'), (g3, '$g3$'), origin=True, shape=(1, 2))
# draw_some((img-g0, '$L0$'), (G1-g1, '$L1$'), (G2-g2, '$L2$'), (G3-g3,'$L3$'), origin=True, shape=(1, 2))
# draw_some((up_resolution(G3, padding_mode=0), '零填充'), (up_resolution(G3, padding_mode=1), '边界环绕'), (up_resolution(G3, padding_mode=2), '边界复制'), (up_resolution(G3, padding_mode=3), '边界镜像'))
# 尝试不同的采样频率
# draw_some((img, '原图'), (sampling(img, 2), '1/2'), (sampling(img, 4), '1/4'), (sampling(img, 8), '1/8'))
# gauss_img = conv(img, gauss_filter(2), padding_mode=3)
# draw_some((gauss_img, '原图'), (sampling(gauss_img, 2), '1/2'), (sampling(gauss_img, 4), '1/4'), (sampling(gauss_img, 8), '1/8'))
# # Gauss一阶微分
# draw_some((gauss_filter(6, derivative=0), 'x轴偏导', 'line'), (gauss_filter(6, derivative=1), 'y轴偏导', 'line'))
# sigmas = [0.5, 1, 2, 5]
# magnitude, orientation, title = [], [], []
# for sigma in sigmas:
#     m, o = img_gradient(img_gray, sigma)
#     magnitude.append(m)
#     orientation.append(o)
#     title.append(f'$\sigma={sigma}$')
# draw_some(*[(magnitude[i], '幅度图'+title[i], 'line') for i in range(len(sigmas))],
#           *[(orientation[i], '方向图'+title[i], 'clip', 'CMRmap') for i in range(len(sigmas))], shape=(2, 4))
# # Canny边缘检测算法
# canny(img_gray, 0.5)
# Harris角点检测
# corner = harris(img_gray, 0.5, 0.05, show_img=img)
sigma = 0.5
img1, img_gray1 = img_open('../figure/corner_test1.png')
corner1 = harris(img_gray1, sigma, 0.05, show_img=img1)
img2, img_gray2 = img_open('../figure/corner_test2.png')
corner2 = harris(img_gray2, sigma, 0.05, show_img=img2)
draw_some((corner1, f'窗口大小${int(4*sigma+1)}\\times{int(4*sigma+1)}$'), (corner2, f'窗口大小${int(4*sigma+1)}\\times{int(4*sigma+1)}$'))
