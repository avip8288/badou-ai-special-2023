# 导入NumPy库，并为其设置别名np
import numpy as np

# 从sklearn.datasets中导入load_iris函数，这个函数可以加载Iris数据集
from sklearn.datasets import load_iris

# 从sklearn.decomposition中导入PCA类，这个类可以用来进行主成分分析
from sklearn.decomposition import PCA

# 导入matplotlib.pyplot模块，这个模块常用于数据可视化
import matplotlib.pyplot as plt

# 加载Iris数据集，数据集中包含150个样本，每个样本有4个特征
# 这些特征包括萼片长度、萼片宽度、花瓣长度和花瓣宽度
# 数据的真实目标值（也就是花的种类）被储存在目标变量y中
# 目标名字被储存在target_names中
data = load_iris()
X = data.data  # 获取特征数据
y = data.target  # 获取目标数据
target_names = data.target_names  # 获取目标名字

# 创建一个PCA对象，设置主成分数量为2，对数据进行降维处理
# 使用fit_transform方法对数据进行拟合和转换，结果储存在X_r中
pca = PCA (n_components=3)
X_r = pca.fit_transform (X)

# 为三个不同种类的花分别设置颜色，并为每种花的线条设置宽度
colors = ['navy' , 'turquoise' , 'darkorange']
lw = 2

# 使用for循环遍历每种花的颜色、线条宽度和目标名字，使用plt.scatter绘制散点图
for color , i , target_name in zip (colors , [0 , 1 , 2] , target_names) :
    plt.scatter (X_r[y == i , 0] , X_r[y == i , 1] , color=color , alpha=.8 , lw=lw , label=target_name)

# 使用plt.legend添加图例，设置图例的最佳位置，关闭阴影，每个类别都有一个图例
plt.legend (loc='best' , shadow=False , scatterpoints=1)

# 设置图的标题为'PCA of IRIS dataset'
plt.title ('PCA of IRIS dataset')

# 显示图形
plt.show ()