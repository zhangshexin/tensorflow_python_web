import cv2
import os
img = cv2.imread(os.getcwd()+'/imgs/mini_zjz.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray.jpg", gray)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite("binary.jpg", binary)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
cv2.putText(img, "{:.3f}".format(len(contours)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1)
# cv2.imshow("img", img)
cv2.imshow("contours", img)
cv2.waitKey(0)

# 原文链接：https: // blog.csdn.net / zong596568821xp / article / details / 81318934