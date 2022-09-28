import matplotlib.pyplot as plt

config = {
    "font.family": 'serif', # 衬线字体
    "figure.figsize": (6, 6),  # 图像大小
    "font.size": 14, # 字号大小
    "font.serif": ['SimSun'], # 宋体
    "mathtext.fontset": 'stix', # 渲染数学公式字体
    'axes.unicode_minus': False # 显示负号
}
plt.rcParams.update(config)
