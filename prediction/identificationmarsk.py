import cv2
import  numpy as np
import os



imgpath=os.getcwd()+'/imgs'

img=cv2.imread(imgpath+'/zjz.jpeg')
img2=np.copy(img)

# '''
#缩放
rows,cols,channels=img.shape
img=cv2.resize(img,None,fx=0.5,fy=0.5)
rows,cols,channels=img.shape
# cv2.imshow('img',img)

#转换hsv
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# lower_blue=np.array([164, 141  ,65])
# upper_blue=np.array([184 ,161 ,145])
pixstart=hsv[0,0]
pixend=hsv[0,cols-1]

lower_blue=np.array([pixstart[0]-10,pixstart[1]-10,pixstart[2]-50])
upper_blue=np.array([pixend[0]+80,pixend[1]+80,pixend[2]+100])
print(lower_blue,upper_blue)
mask=cv2.inRange(hsv,lower_blue,upper_blue)
cv2.imshow('Mask',mask)


#腐蚀-》膨胀，进行开运算，去除小区域黑洞
kernel = np.ones((30, 30), np.uint8)
erode=cv2.erode(mask,kernel)
cv2.imshow('erode',erode)

dilate=cv2.dilate(erode,kernel)
cv2.imshow('dilate',dilate)



#遍历替换
for i in range(rows):
    for j in range(cols):
        if mask[i,j]==255:
            img[i,j]=(255,255,255)#此处替换颜色为bgr通道

cv2.imshow('res',img)

graimg=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',graimg)
_, thresh = cv2.threshold(graimg, 61, 122, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
edges = cv2.Canny(thresh, 30, 70)
cv2.imshow('edges',edges)



#高斯作模糊处理，边缘过于锐化不好看
'''
gauimg=cv2.GaussianBlur(img, ksize=(15, 15), sigmaX=0, sigmaY=0)

cv2.imshow('gau',gauimg)

#将模糊后的图和原图叠加
#alpha，beta，gamma可调
alpha = 0.6
beta = 1-alpha
gamma = 0
img_add = cv2.addWeighted(img, alpha, gauimg, beta, gamma)
cv2.imshow('imgadd',img_add)
'''
cv2.waitKey(0)
cv2.destroyAllWindows()