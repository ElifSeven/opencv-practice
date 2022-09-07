# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:34:49 2022

@author: Elif
"""
import cv2
from matplotlib import pyplot as plt

image = cv2.imread("ucak.jpg",0)

# yeni bir pencerede resmi yeniden boyutlandirmak
cv2.namedWindow("image",cv2.WINDOW_AUTOSIZE)
## cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.imshow("image",image)


cv2.imshow("image window", image)

plt.imshow(image,cmap="gray")
plt.show()

k = cv2.waitKey(0)

if k == 27:
    print("esc tusuna basildi")
    
elif k == ord("s"):
    print("s tusuna basildi")
    cv2.imwrite("griucak.jpg",image)

cv2.destroyAllWindows()