## CVPR Lecture 14

CIFAR10 数据集，数据集大小较小，图像大小为32x32x3（3072）.

### 线性分类器

代数角度：
$$
f(x,W) = Wx+b
$$
视觉角度：将线性分类器 $W$ 的第 $i$ 行，记为 $W_i$，还原成图像，则可以看出来具有该类别的特征，使得具有近似像素的图像具有更高的响应结果.

<img src="./Computer Vision Lecture 14.figure/image-20221026143016601.png" alt="image-20221026143016601" style="zoom:30%;" />

几何角度：决策边界：令参数方程为 $0$. $W_ix+b_i=0$ 就是第 $i$ 种类别的决策边界.

### SVM

#### SVM损失

对于第 $i$ 个样本，$s_{y_i}$ 表示在正确类别 $y_i$ 上的分数，$s_j$ 表示在别的类别上的分数. Hinge损失（活页损失，指损失函数类似一个活页门），数据集的损失为 $L = \frac{1}{n}\sum_{}L_i$ 

<img src="./Computer Vision Lecture 14.figure/image-20221026143742003.png" alt="image-20221026143742003" style="zoom:33%;" />

$W$ 的选择具有多样性，若 $W$ 满足 $L=0$，则 $kW,\ (k\geqslant 1)$ 都是满足 $L = 0$ 的.

#### 交叉熵损失(Softmax)

将分数转为概率（softmax函数）：
$$
P(Y=k|X = x_i) = \frac{e^{s_k}}{\sum_{j}e^{s_j}}
$$
损失函数
$$
L_i = -\log P(Y=y_i|X=x_i)
$$
$L_i$ 的最小值为 $0$（分类完全正确），最大值 $+\infty$（分类错误）.

### 损失函数

$$
L(W) = \frac{1}{N}\sum_{i=1}^NL_i(f(x_i,W),y_i)+\lambda R(W)
$$

第一项 $\frac{1}{N}\sum_{i=1}^NL_i(f(x_i,W),y_i)$ 为数据损失（模型应更接近训练数据），第二项 $\lambda R(W)$ 为正则项（防止模型过拟合于训练数据），$\lambda$ 为正则化强度（超参数）.

#### 正则化函数

1. $L^2$ 正则化：$R(W) = \sum_{k}\sum_{l}W^2_{k,l}$，当有多个 $W$ 具有相同的输出结果，正则化期望参数分布更加离散（光滑）. 例如：$R([1,0,0,0]) = 1,\ R([0.25,0.25,0.25,0.25])=0.25$，所以第二个更加平滑，具有更好的泛化性.
2. $L^1$ 正则化：$R(W) = \sum_{k}\sum_{l}|W_{k,l}|$.