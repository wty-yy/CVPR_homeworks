# -*- coding: utf-8 -*-

"""
@File    : test.py
@Author  : wtyyy
@Time    : 2022/10/5 22:05
"""

import numpy as np
import matplotlib.pyplot as plt
n = 512
im1=np.random.random((n,n,3))
im2=np.random.random((n//2,n//2,3))
im3=np.random.random((50,40,3))
im4=np.random.random((60,50,3))

fig = plt.figure(figsize=(3.41*2, 3.41))
rect1 = [0.01, 0.1, 30/60, 0.8]
rect2 = [40/60, 0.1, 15/60, 0.8]
ax = plt.axes(rect1)
print(type(ax))
print(type(plt.gca()))
plt.imshow(im1)
plt.axis('off')
plt.axes(rect2)
plt.imshow(im2)
plt.axis('off')
plt.show()