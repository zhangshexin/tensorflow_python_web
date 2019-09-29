这是一个基于flask框架的tython web项目，用于练手

maskmode下载地址：https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md



导入
>> import cv2
>> import numpy as np
读图片
>> image_arr = cv2.imread('file_path')
灰度图扩展成彩色图
可以通过图片的channel判断是否是灰度图。如果需要可以将灰度图扩展到RGB的彩色图（复制灰度图的数据到各通道）

>> if image_arr.shape[2] == 1:
      image_arr_rgb = cv2.cvtColor(image_arr, cv2.COLOR_GRAY2RGB)
彩色图像素存储格式
imread 读的彩色图按照BGR像素存储，如果转换成RGB则需要用cvtColor函数进行转换

>> image_arr_rgb = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)
图片size存储格式
imread 读的图片按照 H,W,C 格式存储

>> image_arr_rgb.shape
(H, W, C)
H,W,C格式转换到C,H,W格式

>> image_arr_rgb_chw = np.transpose(image_arr_rgb, (2,0,1))

``````````````````````````````````````

cv2.IMREAD_COLOR : 读入图片,任何与透明度相关通道的会被忽视,默认以这种方式读入.
cv2.IMREAD_GRAYSCALE : 以灰度图的形式读入图片.
cv2.IMREAD_UNCHANGED : 保留读取图片原有的颜色通道.