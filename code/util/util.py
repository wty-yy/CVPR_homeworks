# -*- coding: utf-8 -*-

"""
@File    : util.py
@Author  : wtyyy
@Time    : 2022/9/26 8:33
@Function:
    img_open(fname): 打开图像文件，返回img, img_gray原图与灰度图
    gauss(x, y, sigma, dim): 高斯函数，dim=1为一维高斯，dim=2为二维高斯
    gauss_filter(k, dim=2): 高斯滤波器，大小为(2k+1)x(2k+1)，方差为sigma=k/2的高斯核
    gauss_filter_fixed(n, m, sigma): 固定大小的高斯滤波器，大小为nxm，方差为sigma
    padding(img, dx, dy, mode=0): 原始图像img，横向增加dx，纵向增加dy个像素
        mode=0: 零填充
        mode=1: 边界环绕
        mode=2: 边界复制
        mode=3: 镜像边界
    cov(img, filter, mode=0): 卷积函数，原始图像img, 卷积核filter，mode=0等宽卷积（其他卷积还没用到）
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def img_open(fname):
    img = np.array(Image.open(fname))
    img = img[0:img.shape[0], 0:img.shape[1], :3]
    img_gray = np.expand_dims(np.array(Image.fromarray(img).convert('L')), -1)
    img = img.astype(np.float64) / 256  # 统一为[0,1)的数据范围
    img_gray = img_gray.astype(np.float64) / 256
    return img, img_gray

def img_change(img):
    img_RGB = Image.fromarray(img).convert('RGB')
    img_gray = img_RGB.convert('L')
    return np.expand_dims(np.array(img_RGB), -1).astype(np.float64) / 256, \
           np.expand_dims(np.array(img_gray), -1).astype(np.float64) / 256

pi = np.pi
def gauss(x, y, sigma, dim=2):  # Gauss函数
    sigma_2 = sigma * sigma
    if dim == 2:
        return 1 / (2 * pi * sigma_2) * np.exp(-(x*x + y*y) / (2 * sigma_2))
    return 1 / (np.sqrt(2 * pi) * sigma) * np.exp(-(x*x) / (2 * sigma_2))

def gauss_partial(x, y, sigma, d=0):  # 二维Gauss函数偏导，d为偏导对象
    pos = [x, y]
    return -pos[d] / (2*np.pi*np.power(sigma, 4)) * np.exp(-(x*x + y*y)/(2*sigma*sigma))

def gauss_filter(k, dim=2, derivative=None):  # 返回一个2k+1*2k+1的sigma=k/2的高斯滤波器
    n = 2 * k + 1
    filter = np.zeros([n, n, 1])
    for i in range(n):
        for j in range(n):
            if dim == 1 and i == k:
                filter[i, j, 0] = gauss(j-k, 0, sigma=k/2, dim=1)
            elif dim == 2:
                filter[i, j, 0] = gauss(i-k, j-k, k/2)
            if derivative is not None:
                filter[i, j, 0] = gauss_partial(i - k, j - k, k / 2, d=derivative)
    if derivative is not None:  # 偏导结果，无需正规
        return filter
    return filter / np.sum(filter)

def gauss_filter_fixed(n, m, sigma):
    mid_n, mid_m = n // 2, m // 2
    filter = np.zeros([n, m, 1])
    for i in range(n):
        for j in range(m):
            filter[i, j, 0] = gauss(i-mid_n, j-mid_m, sigma)
    return filter / np.sum(filter)

def padding(img, dx, dy, mode=0):  # 原始图像img，横向增加dx，竖向增加dy个像素
    # mode = 0: 零填充
    # mode = 1: 边界环绕
    # mode = 2: 边界复制
    # mode = 3: 镜像边界
    n, m, c = img.shape
    lx = (dx + 1) // 2  # 左侧填充量(向上取整)
    uy = (dy + 1) // 2  # 上侧填充量(向上取整)
    new_img = np.zeros([n + int(dx), m + int(dy), c])
    for i in range(n + dx):
        for j in range(m + dy):
            id1 = -1 if i < lx else (1 if i >= n + lx else 0)
            id2 = -1 if j < uy else (1 if j >= m + uy else 0)
            # 将原始图像分为9个区域，编号为中间区域坐标为(0,0)，左上角区域坐标为(-1,-1)
            # 除中间区域(0,0)外，其他区域为填充区域，根据不同要求进行填充
            if (id1, id2) == (0, 0):
                new_img[i, j, :] = img[i - lx, j - uy, :]
                continue
            if mode == 0:  # 零填充
                new_img[i, j, :] = np.zeros(c)
            elif mode == 1:  # 边界环绕，将图像进行平移
                xx, yy = -id1 * n, -id2 * m
                new_img[i, j, :] = img[i + xx - lx, j + yy - uy, :]
            elif mode == 2 or mode == 3:
                # 边界复制，直接拷贝边界处的像素值
                # 边界镜像，以边界作为对称轴，两种方法都需要找到基准点
                def symmetric(x, o):  # 对称中心o，当前向量x
                    x = np.array(x)
                    o = np.array(o)
                    x1, x2 = (2 * o - x).tolist()
                    return img[x1, x2, :]
                if id1 * id2 != 0:
                    id1 = max(0, id1) * (n-1)
                    id2 = max(0, id2) * (m-1)
                    if mode == 2:
                        new_img[i, j, :] = img[id1, id2, :]
                    else:
                        new_img[i, j, :] = symmetric((i-lx, j-uy), (id1, id2))
                else:
                    if id1 == 0:
                        id2 = max(0, id2) * (m-1)
                        if mode == 2:
                            new_img[i, j, :] = img[i - lx, id2, :]
                        else:
                            new_img[i, j, :] = symmetric((i-lx, j-uy), (i-lx, id2))
                    else:
                        id1 = max(0, id1) * (n-1)
                        if mode == 2:
                            new_img[i, j, :] = img[id1, j - uy, :]
                        else:
                            new_img[i, j, :] = symmetric((i-lx, j-uy), (id1, j-uy))
    return new_img

def conv(img, filter, mode=0, stride=1, padding_mode=0):
    if img.ndim == 2:
        img = np.expand_dims(img, -1)
    n, m, c = img.shape
    a, b = filter.shape[0:2]
    shape = np.array(img.shape).astype(int)
    shape[0:2] //= stride
    output = np.zeros(shape)
    if mode == 0:
        img = padding(img, a-1, b-1, padding_mode)
    for i in tqdm(range(0,n,stride)):
        for j in range(0,m,stride):
            for k in range(c):
                output[i//stride, j//stride, k] = np.sum(img[i:i+a, j:j+b, k] * filter[:,:,0])
    return output

def conv1(img, filter, mode=0):
    n, m, o = img.shape
    a, b = filter.shape[0:2]
    filter = filter.tolist()
    output = np.zeros_like(img).astype(np.float64).tolist()
    if mode == 0:
        img = padding(img, a-1, b-1)
    img = img.tolist()
    for i in range(n):
        for j in range(m):
            for k in range(o):
                for u in range(a):
                    for v in range(b):
                        output[i][j][k] += img[i+u][j+v][k] * filter[u][v][0]
    return np.array(output)

