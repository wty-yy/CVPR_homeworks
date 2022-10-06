# -*- coding: utf-8 -*-

"""
@File    : filter_fourier.py
@Author  : wtyyy
@Time    : 2022/10/6 12:27
@Function:
    DOG(img, sigma1, sigma2): 返回DOG核，sigma1,2为两个Gauss核的方差，
                              使用小方差减去大方差即得到DOG核.
    sharp_filter(alpha, sigma): 返回锐化核，(1+alpha)I-alpha*gauss，
                                其中I为恒等变换，gauss为方差为sigma的Gauss核.
    bilateral(img, sigma1, sigma2, show=[]): 双边滤波器，
                                sigma1为对空域使用的Gauss核，sigma2对值域使用的Gauss核方差
                                show中为像素坐标，作用是将该处对应的滤波核打印出来.
    fft(img): 返回值: 平移后的频域，逆变换，幅度谱，相位谱.
    combine_magnitude_phase(magnitude, phase): 返回由幅度谱和相位谱构成的频率.
    ft_circle(ft_shift, r): 以半径为r的圆，将频域图划分为两部分，内圈与外圈
                            返回: 内圈频域，外圈频域，内圈逆变换，外圈逆变换
    show_freq(img): 返回经过np.log(np.abs(img))加先行正规化后的图像，用于可视化频域图.
"""

from tqdm import tqdm
from util.draw_figure import draw_some
from util.util import *

def DOG(img, sigma1, sigma2):
    if sigma1 > sigma2:
        sigma1, sigma2 = sigma2, sigma1
    k1, k2 = np.ceil(sigma1 * 2).astype(int), np.ceil(sigma2 * 2).astype(int)
    gauss1, gauss2 = gauss_filter(k1), gauss_filter(k2)
    return conv(img, gauss1) - conv(img, gauss2)

def sharp_filter(alpha, sigma):
    k = np.ceil(sigma * 2).astype(int)
    gauss = gauss_filter(k)
    I = np.zeros([2*k+1, 2*k+1, 1])
    I[k, k, 0] = 1
    return (1 + alpha) * I - alpha * gauss

def bilateral(img, sigma1, sigma2, show=[]):
    mean = img.mean()
    K = sigma1 * 2  # 根据高斯滤波器算滤波器大小
    filter1 = gauss_filter(K)[:, :, 0]
    n, m, o = img.shape
    a, b = filter1.shape[0:2]
    output = np.zeros_like(img)
    img = padding(img, a-1, b-1)
    for i in tqdm(range(n)):
        for j in range(m):
            for k in range(o):
                small_img = img[i:i+a, j:j+b, k]
                gauss_line = lambda x: gauss(x, 0, sigma2, dim=1)
                filter2 = gauss_line(small_img - img[i+K, j+K, k])
                new_filter = filter1 * filter2
                new_filter /= np.sum(new_filter)
                output[i, j, k] = np.sum(small_img * new_filter)
                if (i, j) in show:
                    draw_some((small_img, '原图'), (new_filter, '滤波器'), (filter2, '值域滤波'))
    return output

def fft(img):
    img = img.reshape([img.shape[0], -1])
    ft = np.fft.fft2(img)
    ft_shift = np.fft.fftshift(ft)
    ift = np.abs(np.fft.ifft2(ft))
    magnitude = np.abs(ft_shift)  # 幅度谱
    phase = np.angle(ft_shift)  # 相位谱
    return [ft_shift, ift, magnitude, phase]

def combine_magnitude_phase(magnitude, phase):
    ft = magnitude * np.exp(1j * phase)
    ift = np.abs(np.fft.ifft2(ft))
    return ift

def ft_circle(ft_shift, r):
    in_circle, out_circle = ft_shift.copy(), ft_shift.copy()
    n, m = ft_shift.shape
    for i in range(n):
        for j in range(m):
            if np.linalg.norm((i-n/2, j-m/2)) < r:
                out_circle[i, j] = 0
            else:
                in_circle[i, j] = 0
    return in_circle, out_circle, np.fft.ifft2(in_circle), np.fft.ifft2(out_circle)

def show_freq(img):
    img = np.log(np.abs(img))
    img = (img - img.min()) / (img.max() - img.min())
    return img
