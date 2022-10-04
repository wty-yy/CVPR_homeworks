# Computer Vision Lecture 1

**Computer Vision(CV)**: The study of how **computer** can be **programmed** to **extract useful information** about the environment from **optical images**.

## Input

### Gray-Scale

A (gray-scale) image is a 2D array. For more, any 2D array of real numbers can be treated as an image. (white=high, black=low)

### Color image

width $\times$ height $\times$ channels

## Output

Depends on what we want to do with the image.

The goal of computer vision:

- Reconstruction: Understanding 3D structure of the world
- Grouping/Re-organization(Unsupervised):  Group pixels into objects
- Recognition(Supervised): Classify objects, scenes, actions...

## 5R Problems in CV

1. Registration: 配准问题
2. Reconstruction: 重建问题
3. Representation: 表示问题，invariance不变形，equivariance等变形，discriminability辨识能力
4. Re-organization: 重组问题，pixels to X 模式识别，无监督学习
5. Recognition: 识别问题，classification/detection

![image-20220907161538308](Computer Vision Lecture 1.figure/5R Problems)

## Course Overview

1. Low/mid-level vision: 滤波器（信号处理），图像形成，图像结构
2. Reconstruction
3. Recognition: 传统机器学习，深度神经网络

## Project

1. Feature detection and matching
2. Camera calibration
3. Feature Tracking(Sparse optical flow)
4. Deep learning for classification & detection

## Conferences

http://conferences.visionbib.com/Iris-Conferences.html

