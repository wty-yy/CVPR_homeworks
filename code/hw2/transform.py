# -*- coding: utf-8 -*-

"""
@File    : transform.py
@Author  : wtyyy
@Time    : 2022/10/4 18:00
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.split(sys.path[0])[0])
from util.util import *
from util.draw_figure import draw_some

img, img_gray = img_open('../figure/fox2.png')
draw_some((img, '原图'), (padding(img, 200, 200, mode=0), '零填充'),
          (padding(img, 200, 200, mode=1), '边界环绕'),
          (padding(img, 200, 200, mode=2), '边界复制'),
          (padding(img, 200, 200, mode=3), '边界镜像'))