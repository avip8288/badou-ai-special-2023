# -*- coding: utf-8 -*-  
__author__ = '李元豪 from https://www.zhilu.space'

# 导入cv2库，cv2是OpenCV的Python版本
import cv2

# 导入matplotlib.pyplot库，用于图像的显示和操作
import matplotlib.pyplot as plt

# 使用cv2的imread方法读取名为'lenna.png'的图片，并将其转化为灰度图像
# 'lenna.png'可能是一个灰度图像的路径，这取决于实际环境和你的代码文件结构
img = cv2.imread (r"D:\AI1026\a22.png", 0)

# 将图像从BGR颜色空间转换为RGB颜色空间，因为在OpenCV中默认读取的图像是BGR格式的
# 这一步是为了后面使用cv2.cvtColor方法进行颜色空间转换时不会报错
img_gray = cv2.cvtColor (img , cv2.COLOR_BGR2RGB)

# 使用高斯滤波对图像进行模糊处理，滤波器的大小为3*3，标准差设置为0，表示根据图像自动计算
img_blur = cv2.GaussianBlur (img_gray , (3 , 3) , 0)

# 使用Sobel算子计算x方向上的梯度，即水平方向的边缘强度
xgrad = cv2.Sobel (img_blur , cv2.CV_16SC1 , 1 , 0)

# 使用Sobel算子计算y方向上的梯度，即垂直方向的边缘强度
ygrad = cv2.Sobel (img_blur , cv2.CV_16SC1 , 0 , 1)

# 使用Canny边缘检测算法，利用之前计算得到的xgrad和ygrad来检测边缘，阈值范围为50~150
edge1 = cv2.Canny (xgrad , ygrad , 50 , 150)

# 直接使用高斯滤波后的图像进行Canny边缘检测，阈值范围同样为50~150
edge2 = cv2.Canny (img_blur , 50 , 150)

# 使用imshow方法显示原始图像
cv2.imshow ('origin image' , img)

# 使用imshow方法显示通过两种不同方式处理后得到的边缘检测图像edge1
cv2.imshow ('edge image' , edge1)

# 使用imshow方法显示通过另一种方式处理后得到的边缘检测图像edge2
cv2.imshow ('edge image2' , edge2)

# 使用waitKey方法，该方法会暂停程序的执行，直到用户关闭所有图像窗口或按任意键继续。这使得图像可以保持显示状态，等待用户操作。
cv2.waitKey ()