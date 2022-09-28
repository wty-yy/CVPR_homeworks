# -*- coding: utf-8 -*-

"""
@File    : circle_test.py
@Author  : wtyyy
@Time    : 2022/9/25 8:49
"""

import numpy as np
import matplotlib.pyplot as plt


def complex_e(x):
    x *= 2 * pi
    return complex(np.cos(x), np.sin(x))


pi = np.pi
space = np.linspace(0, 4.5, 1000)
f = 3  # Hz
u = lambda x: np.cos(2 * pi * f * x) + np.cos(2 * pi * 1 * x)
fig = plt.figure(figsize=(10, 3))
plt.plot(space, u(space))
plt.show()

plt.figure(figsize=(5, 5))
def draw_complex(target, draw=True):
    center = 0
    x, y = [], []
    for i in np.linspace(0, 4.5, 1000):
        tmp = u(i) * complex_e(-i * target)
        center += tmp
        x.append(tmp.real)
        y.append(tmp.imag)
    center /= 1000
    if draw:
        plt.plot(x, y, label=f'{target=}')
        plt.scatter(center.real, center.imag, label='center')
    return center
target = 0
x, y = np.linspace(0, 5, 200), []
for target in x:
    center = draw_complex(target, False)
    y.append(center.real)
    target += 0.05
#plt.legend(loc='upper right')
#plt.show()

plt.figure(figsize=(10, 3))
plt.plot(x, y)
plt.show()

# fft
N = 1024
M = 5
space = np.linspace(0, 1, N)
wave = u(space)
u_hat = np.fft.fft(wave)
plt.plot(range(M), [u_hat[i].real for i in range(M)])
plt.xticks(range(M))
plt.show()
