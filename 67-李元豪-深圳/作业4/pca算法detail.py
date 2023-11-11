import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 读取图片
img = Image.open (r"D:\AI1026\a22.png")
img_array = np.array (img)

# 数据中心化
mean = np.mean (img_array)
img_array = img_array - mean

# 计算协方差矩阵
cov = np.cov (img_array.T)

# 计算特征值和特征向量
eigenvalues , eigenvectors = np.linalg.eig (cov)

# 对特征向量进行排序
idx = eigenvalues.argsort ()[: :-1]
eigenvectors = eigenvectors[: , idx]

# 选择前k个特征向量
k = 2
eigenvectors_k = eigenvectors[: , :k]

# 将数据投影到新的空间中
new_img = np.dot (img_array , eigenvectors_k)

# 显示原始图片和降维后的图片
fig , axs = plt.subplots (1 , 2)
axs[0].imshow (img_array)
axs[0].set_title ('Original Image')
axs[1].imshow (new_img)
axs[1].set_title ('PCA Image')
plt.show ()