# Computer Vision Lecture 2

## 图片的定义

由像素矩阵构成，0为黑，255为白.

像素值：

1. 灰度值 $[0,255]$.

2. 彩色值：
    - RGB [R, G, B]：每个通道数字范围为 $[0,255]$.
    - Lab [L, a, b]：L为光强度，a和b为色彩分量维度（color-opponent dimensions）
    - HSV [H, S, V]：Hue, saturation, value

### 图像函数

将图像视为函数：$f\mathbb{R}^2\to\mathbb{R}\text{ or }\mathbb{R}^M$.

- 灰度：$f(x, y)$ 表示 $(x, y)$ 处像素的灰度值.
- 彩色：$f(x, y) = [r(x, y), g(x, y), b(x, y)]$.

### 图像变换

将图像映射到图像的变换，设 $f(x, y)$ 为原始图像，

如：$g(x, y)  = f(x, y)+20$，$g(x, y) = f(-x, y)$.

图像去噪，图像超分辨（低分辨率转高分辨率，还原细节）.

图像噪声的产生：传感器噪声（光信号转化为电信号），坏掉的像素（数字存储问题），老照片

对于噪声的假设条件：每个噪声的产生具有**独立性**，噪声的分布满足Gauss分布.

**简单去噪方法**：由于大部分相近的像素具有相同的颜色，用周围像素的均值代替原像素.

## 滤波器

### 均值滤波器（模糊处理 Blur）

滑动滤波器窗口，将每个滤波器像素结果作为新图像的像素.

作用：提取有效信息，增强图片（模糊化图像去噪，锐化图像）

设 $f$ 原有图像： $9\times 9$ 均值滤波器，滤波算子 $S$ 定义如下.
$$
S[f](m, n) = \sum_{i=-1}^1\sum_{j=-1}^1f(m+i, n+j)/9
$$
更一般的滤波器（核函数） $w(i, j)$：
$$
S[f](m, n) = \sum_{i=-1}^1\sum_{j=-1}^1w(i,j)f(m+i, n+j)
$$
更一般的滤波器大小（核大小） $(2k+1)\times (2k+1)$：
$$
S[f](m, n) = \sum_{i=-k}^k\sum_{j=-k}^kw(i,j)f(m+i, n+j)
$$
$w(i, j) = 1/(2k+1)^2$ ，均值滤波器

$w(i, j)\geqslant 0$ 且 $\sum w(i, j) = 1$，加权平均，$w(i, j)$ 可以是任意实数

### 图像增强

#### 锐化（增强边界细节 Sharpen）

$$
\begin{aligned}
f_{sharp} =&\ f+\alpha(f-f_{blur})\\
=&\ (1+\alpha)f-\alpha f_{blur}\\
=&\ (1+\alpha)(w*f)-\alpha(v*f)\\
=&\ ((1+\alpha)w-\alpha v)*f
\end{aligned}
$$

其中 $w$ 为原图像滤波器，$v$ 为均值滤波器.

### 其他滤波器

如下 $11\times 11$ 的滤波器，抽取具有类似结构的图像.

<img src="Computer Vision Lecture 2.figure/一种特殊的滤波器" alt="image-20220909175357157" style="zoom:50%;" />

**非线性滤波器**：

- $$
    g(m, n) = \begin{cases}
    255,&\quad f(m, n) > A,\\
    0,&\quad \texttt{otherwise.}
    \end{cases}
    $$
    
- $$
    g(m, n) = \max(f(m, n), 0)
    $$
    
- 中值滤波器，将原滤波器核区域进行排序然后取中间值作为代替：<img src="Computer Vision Lecture 2.figure/中值滤波器" alt="image-20220909183103480" style="zoom:50%;" />

滤波器设计：

<img src="Computer Vision Lecture 2.figure/几种滤波器" alt="image-20220909183219766" style="zoom:50%;" />




## 卷积与互相关

卷积 Convolution 和互相关  Cross-correlation.

###  性质

> 互相关
> $$
> S[f] = w\otimes f\\
> S[f](m, n) = \sum_{i=-k}^k\sum_{j=-k}^kw(i, j)f(m+i, n+j)
> $$
> 卷积
> $$
> S[f] = w* f\\
> S[f](m, n) = \sum_{i=-k}^k\sum_{j=-k}^kw(i, j)f(m-i, n-j)
> $$
> 

将 $w$ 旋转 $180$ 度（横向旋转再竖向旋转）得 $w'$，则
$$
w\otimes f = w'*f
$$
**线性性**： 设 $a,b\in \mathbb{R}$，

$f' = af+bg$，$w\otimes f' = a(w\otimes f)+b(w\otimes g)$.

$w' = aw+bv$，$w'\otimes f = a(w\otimes f)+b(v\otimes f)$.

**平移不变性**：$f'(m, n) = f(m-m_0, n-n_0)$，则
$$
\begin{aligned}
(w\otimes f')(m, n) =&\ \sum_{i=-k}^k\sum_{j=-k}^kw(i, j)f'(m+i,n+j)\\
=&\ \sum_{i=-k}^k\sum_{j=-k}^kw(i, j)f(m+i-m_0,n+j-n_0)\\
=&\ (w\otimes f)(m-m_0, n-n_0)
\end{aligned}
$$
可用于像素标记，假设 $I$ 为图像，$\phi(I)$ 是一个标记函数，$T(\cdot)$ 为平移函数，则
$$
\phi(T(I)) = T(\phi(I))
$$
**对称性**：$w * f = f*w$.

**结合律**：$v*(w*f) = (v*w)*f$.

**等变性**：设图像 $I$ 的变换群（全体变换组成的集合）为 $T$，一个变换 $\phi:I\to F$，对任意的图像 $I_1$ 都存在等变换 $S:F\to F$ 使得
$$
S[\phi(I_1)] = \phi[T(I_1)]
$$
<img src="Computer Vision Lecture 2.figure/等变性" alt="image-20220909181154408" style="zoom:50%;" />

**不变性**：

**模板识别（图像相关性）**：从一张图像中寻找另一个图像（模板）的位置.

用两个图像做内积（互相关）可以得到两个图像的近似度，故**将图像模板作为滤波器核**作用在原图上，做互相关后亮度较高的位置就是原图和模板最相关的图像.

## 卷积的分类

### 全卷积(Full convolution)

计算每一个核与原图像相交的位置. 输出大小 $m+k-1\times m+k-1$.

### 相同卷积（Same convolution）

保持输入图像和输出图像相同. 输出大小为 $m\times m$.

### 有效卷积（Valid convolution）

仅计算有图像的位置（不进行填补）. 输出大小为 $m-k+1\times m-k+1$.

## 填补方法

1. 补零
2. 环绕
3. 对称边界
4. 边界复制

## Gauss滤波器

将三维Gauss对以原点为中心，取卷积核大小作映射，然后归一化处理即可. （卷积核大小不能太大，大小取在Guass函数在 99% 以内，例如 $3\sigma$ 原则）

### 分解卷积核（提高计算速度）

分解卷积核为列成行的形式：
$$
w*I = (u*v)*I = u*(v*I)
$$
原复杂度：$O(whk^2)$，分解后复杂度：$O(2whk)$.

## 混合滤波器

利用竟可能小的核可以提高计算速度，而且大的卷积核可以分解为多个小的卷积核. 需要消耗更多的内存.

## 多通道滤波器

将一组滤波器作用在图像上，得到一组图像，一组图像称为特征向量.
$$
[w_1,\cdots, w_k]*I=[F_1,\cdots, F_k]
$$

## 3维卷积（互相关）

$$
w*I(m,n,c) = \sum_{i,j,k}w(i, j, k)I(m+i, n+j, c+k)
$$

### Gauss滤波器

$$
GB[I]_p = \sum_{q\in S}G_{\sigma_s}(||p-q||)I_q
$$

### 双边滤波器

$$

BF[I]_p = \frac{1}{W_p}\sum_{q\in S}G_{\sigma_s}(||p-q||)G_{\sigma_r}(|I_p-I_q|)I_q
$$

保护边缘（Gauss滤波器无法做到），距离差距与像素差距的Gauss函数值进行内积求和.

多次重复做双边滤波器会产生卡通效果，图像变得更加平滑，产生色块.

问题：无法加速计算.

关注相似对象.
