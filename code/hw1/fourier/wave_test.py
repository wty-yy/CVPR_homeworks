# -*- coding: utf-8 -*-

"""
@File    : wave_test.py
@Author  : wtyyy
@Time    : 2022/9/24 23:00
"""
import numpy as np
import matplotlib.pyplot as plt

f = lambda x: np.sin(x) + np.sin(2*x) + np.sin(5*x)
pi = np.pi

N = 1024  # 频域
x = np.linspace(0, 2*pi, N)
wave = f(x)
fig = plt.figure(figsize=(10, 5))
plt.plot(x, wave)
fig.tight_layout()
# plt.show()


M = 20
transformed = np.fft.fft(wave)
print(transformed[0].real)
fig = plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
print(transformed[0:20].real)
plt.plot(range(M), [transformed[i].real for i in range(M)])
plt.xticks(range(M))
fig.tight_layout()
plt.show()

a = []
for i in range(N):
    if transformed[i] > 0.1:
        a.append(i)

print(a)