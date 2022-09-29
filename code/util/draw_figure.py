# -*- coding: UTF-8 -*-

"""
@author: wty-yy
@software: PyCharm
@file: draw_figure.py
@time: 2022/9/25 14:02
"""
import matplotlib.pyplot as plt
import numpy as np

def draw(ax, img, title, norm):
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
    cmap = 'gray' if img.shape[2] == 1 else None
    ax.imshow(img, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

def draw_some(*arg, shape=None):
    n = len(arg)
    if shape is None:
        shape = [1, n]
    fig, axes = plt.subplots(*shape, figsize=(3*shape[1], 3*shape[0]))
    axes = np.array(axes).reshape([-1])
    for i in range(n):
        draw(axes[i], *(list(arg[i]) + (3 - len(arg[i])) * [None]))
    fig.tight_layout()
    fig.show()
