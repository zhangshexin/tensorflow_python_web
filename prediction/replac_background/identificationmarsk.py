import cv2
import  numpy as np
import os
import sys


def convertBGD(path,cm):

    img=cv2.imread(path)
    print(img)
    # '''
    #缩放
    rows,cols,channels=img.shape
    img=cv2.resize(img,None,fx=0.5,fy=0.5)
    rows,cols,channels=img.shape
    # cv2.imshow('img',img)

    #转换hsv
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    if cm=='br' or cm=='bw':
        lower_blue=np.array([78,43,46])
        upper_blue=np.array([110,255,255])
    if cm=='rb'or cm=='rw':
        lower_blue=np.array([-10,242,152])
        upper_blue=np.array([189,265,235])
    if cm=='wr'or cm=='wb':
        lower_blue=np.array([-10,-10,215])
        upper_blue=np.array([160 , 14 ,295])
    mask=cv2.inRange(hsv,lower_blue,upper_blue)
    cv2.imshow('Mask',mask)



    #腐蚀-》膨胀，进行开运算，去除小区域黑洞
    kernel = np.ones((3, 3), np.uint8)
    erode=cv2.erode(mask,kernel)
    cv2.imshow('erode',erode)

    dilate=cv2.dilate(erode,kernel)
    cv2.imshow('dilate',dilate)



    #遍历替换
    for i in range(rows):
        for j in range(cols):
            if dilate[i,j]==255:
                img[i,j]=(48,37,100)#此处替换颜色为bgr通道

    cv2.imshow('res',img)
    #高斯作模糊处理，边缘过于锐化不好看
    # gauimg=cv2.GaussianBlur(img, ksize=(15, 15), sigmaX=0, sigmaY=0)
    #
    # cv2.imshow('gau',gauimg)

    #将模糊后的图和原图叠加
    #alpha，beta，gamma可调
    # alpha = 0.6
    # beta = 1-alpha
    # gamma = 0
    # img_add = cv2.addWeighted(img, alpha, gauimg, beta, gamma)
    # cv2.imshow('imgadd',img_add)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args=sys.argv
    path=args[1]
    cm=args[2]
    #cm=bw blue->white
    #cm=br blue->red
    # cm=wb white->blue
    # cm=wr white->red
    # cm=rw red->white
    # cm=rb red->blue
    convertBGD(path,cm)