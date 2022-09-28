# -*- coding: UTF-8 -*-

"""
@author: wty-yy
@software: PyCharm
@file: GaussianFilter.py
@time: 2022/9/25 12:58
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.split(sys.path[0])[0])  # 将上级目录加入到path中
from util.draw_figure import draw, draw_some
from util.util import gauss, padding, conv, img_open

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

def DOG(img, sigma1, sigma2):
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
    magnitude = np.abs(ft_shift)
    phase = np.angle(ft_shift)
    return [ft, ift, magnitude, phase]

def combine_magnitude_phase(magnitude, phase):
    ft = magnitude * np.exp(1j * phase)
    ift = np.abs(np.fft.ifft2(ft))
    return ift

# img, img_gray = img_open('building.png')
# img, img_gray = img_open('CrowTower_mini.png')
img, img_gray = img_open('fox2.png')
# 零填充效果
# draw_some((img, '原图'), (padding(img, 100, 100), '零填充'))

# 高斯核作用效果
# draw_some((img, '原图'), *[(conv(img, gauss_filter(k)), f'$\sigma={k/2},size={2*k+1}\\times {2*k+1}$') for k in range(1,10,3)])

# 两个高斯核卷积
# gauss = [gauss_filter(i) for i in range(6, 11, 2)]
# conv_gauss = padding(conv(gauss[1], gauss[0]), 4, 4)
# draw_some(*[(gauss[i], f'$\sigma={i+3}$') for i in range(3)], (conv_gauss, '$\sigma=3,4$的卷积结果'))
# print(np.abs(conv_gauss[8:12, 8:12, 0] - gauss[2][8:12, 8:12, 0]))

# 高斯核可分离
# k = 6
# vector_gauss = gauss_filter(k, dim=1)
# vector_gauss_T = np.expand_dims(vector_gauss[:, :, 0].T, 2)
# draw_some((vector_gauss_T, f'一维$\sigma={k/2}$'), (vector_gauss, f'一维$\sigma={k/2}$'), (conv(vector_gauss_T, vector_gauss), '卷积结果'), (gauss_filter(k), f'直接计算$\sigma={k/2}$'))

# DOG处理，不同高斯核处理效果相减
# draw_some((img_gray, '原图'), (DOG(img_gray, 1, 2), f'$DOG,\sigma_1={1},\sigma_2={2}$'),
#          (DOG(img_gray, 0.5, 5), f'$DOG,\sigma_1={0.5},\sigma_2={5}$'))

# 锐化滤波器
# blur = conv(img_gray, gauss_filter(6))
# delta = img_gray - blur
# delta1 = delta - delta.min() + 0.5
# delta1[delta1 > 1] = 1
# delta += 0.5
# delta = (delta - delta.min()) / (delta.max() - delta.min())
# draw_some(img_gray, (blur, f'$\sigma=3$'), (delta, '线性正规化'), (delta1, '截断正规化'))
# alpha = [0.5, 1, 2]
# sigma = [6, 6, 6]
# draw_some((img_gray, '原图'), *[(conv(img_gray, sharp_filter(i[0], i[1])), f'$\\alpha={i[0]},\sigma={i[1]}$') for i in zip(alpha, sigma)])
# draw_some((img_gray, '原图'), (conv(img_gray, sharp_filter(1, 6)), f'$\\alpha=1,\sigma=6$'))

# 双边滤波
# noise = np.clip(img_gray + np.random.normal(0, 0.1, img_gray.shape), 0, 1)
# show_point = [(141, 260), (297, 261)]
# show_rock = [(66, 26), (186, 141)]
# draw_some((img_gray, '原图'), (noise, 'Gauss噪声'), (bilateral(img_gray, 6, 0.1, show_point), '双边滤波'))
# draw_some((img, '原图'), (bilateral(img, 6, 0.1), f'$\sigma_s={6},\sigma_r={0.1}$'))
# sigma1 = [2, 6, 18]
# sigma2 = [0.1, 0.25, 100]
# imgs = []
# for i in sigma1:
#     for j in sigma2:
#         imgs.append((bilateral(img_gray, i, j), f'$\sigma_s={i},\sigma_r={j}$'))
# draw_some(*imgs, shape=[3, 3])

# Fourier变换
fft1 = fft(img_gray)
draw_some((img_gray, '原图'), (np.log(fft1[2]), '幅度谱', 'line'), (np.abs(fft1[3]), '相位谱', 'line'), (fft1[1], 'Fourier逆变换'))
img2 = img_open('fox1.png')[1]
fft2 = fft(img2)
# draw_some((img2, '原图'), (np.log(fft2[2]), '幅度谱', 'line'), (np.abs(fft2[3]), '相位谱', 'line'), (fft2[1], 'Fourier逆变换'))
draw_some((img_gray, '图1'), (combine_magnitude_phase(fft1[2], fft2[3]), '图1的幅度与图2的相位'),
          (combine_magnitude_phase(fft2[2], fft1[3]), '图2的幅度与图1的相位'), (img2, '图2'))
