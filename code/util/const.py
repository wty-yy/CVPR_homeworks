# -*- coding: utf-8 -*-

"""
@File    : const.py
@Author  : wtyyy
@Time    : 2022/10/13 12:33
@Const:
"""
import numpy as np

dx = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
dy = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
direction = [None for _ in range(10)]
direction[9] = [np.array([dx[i], dy[i]]) for i in range(9)]
direction[4] = [direction[9][i] for i in range(1, 9, 2)]
direction[8] = []
for i in range(9):
    if i != 5:
        direction[8].append(direction[9][i])