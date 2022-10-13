# -*- coding: utf-8 -*-

"""
@File    : test.py
@Author  : wtyyy
@Time    : 2022/10/5 22:05
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from queue import PriorityQueue

img = cv2.imread('../figure/fox1.png')
edges = cv2.Canny(img,100,200)
edges2 = cv2.Canny(img,50,200)
plt.figure(figsize=(15, 5))
plt.subplot(131),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image1'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(edges2,cmap = 'gray')
plt.title('Edge Image2'), plt.xticks([]), plt.yticks([])
plt.savefig('cv_edge.png', dpi=600)
plt.show()
