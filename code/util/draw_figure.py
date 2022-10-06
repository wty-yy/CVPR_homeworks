# -*- coding: UTF-8 -*-

"""
@author: wty-yy
@software: PyCharm
@file: draw_figure.py
@time: 2022/9/25 14:02
"""
import matplotlib.pyplot as plt
import numpy as np

def draw(ax, img, title, norm, cmap=None):
    if img.ndim == 2:
        img = np.expand_dims(img, -1)
    if img.min() < 0 or img.max() > 1:
        if norm is None:
            img = img - img.min()
            img[img > 1] = 1
        elif norm == 'line':
            if img.min() == -np.inf:
                img[img < 0] = 0
            img = (img - img.min()) / (img.max() - img.min())
        elif norm == 'clip':
            img = img.clip(0, 1)
        elif norm == 'upper' and img.mean() < 0.3:
            mean = img.mean()
            plt.hist(img.reshape([-1]))
            img[img > 0.005] += 0.5 - img.mean()
            img[img > 1] = 1
    if cmap is None:
        cmap = 'gray' if img.shape[2] == 1 else None
    ax.imshow(img, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

def draw_some(*arg, shape=None, origin=False):
    n = len(arg)
    if shape is None:
        shape = [1, n]
    wide, height = 3, 3  # 图像输出最小大小不能小于300px
    sum_wide = 0
    fact = 150
    for i in range(n):
        h, w = arg[i][0].shape[0:2]
        height = max(height, h/fact)
        wide = max(wide, w/fact)
        sum_wide += w
    figsize = (wide * shape[1], height * shape[0])
    if origin:  # 根据原始图像大小进行绘图，只能绘制一行
        fig = plt.figure(figsize=figsize)
        now_wide, space, sum_fig = 0.05, 0.1 / n, 0.8
        for i in range(n):
            k = arg[i][0].shape[1] / sum_wide
            ax = plt.axes([now_wide, 0.05, k * sum_fig, 0.8])
            draw(ax, *(list(arg[i]) + (3 - len(arg[i])) * [None]))
            now_wide += k * sum_fig + space
    else:
        fig, axes = plt.subplots(*shape, figsize=figsize)
        axes = np.array(axes).reshape([-1])
        for i in range(n):
            draw(axes[i], *(list(arg[i]) + (3 - len(arg[i])) * [None]))
        fig.tight_layout()
    fig.savefig('tmp.png', dpi=600)
    fig.show()
