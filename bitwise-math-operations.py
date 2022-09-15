# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 13:47:40 2022

@author: Elif
"""

import cv2
import numpy as np

x = np.uint8([250])
y = np.uint8([10])

'''
bit alma işlemi olduğu için 260%256 ya gore modunu alıyor
'''
sonuc1 = x+y 

'''
opencv ile toplama islemi yaptığımızda max 255 oluyor. Böylece beyaz tona geciyor
'''
sonuc2 = cv2.add(x,y)


'''
IMAGE EKLEME ISLEMLERI
'''
img1 = cv2.imread("road.jpg")
img2 = cv2.imread("d.jpg")

img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))


total = cv2.addWeighted(img1,0.3,img2_resized,0.7,0) # image baskınlık yüzdeleri
# cv2.imshow("image",total)

'''
Bit Operations
'''
'''
ilk olarak gri tonlamaya donusturduk, daha sonra belirli değerin altındakileri siyah geri kalanları beyaz yaptık.
 10 degerini belirledik.
'''
img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
ret,mask = cv2.threshold(img1_gray,10,255,cv2.THRESH_BINARY)
# cv2.imshow("image",img1_gray)

'''
Kırpma islemleri
'''

x,y,z = img1.shape
roi = img2_resized[0:x,0:y]
# cv2.imshow("image",roi)


'''
bitwise operations on images
'''
mask_inv = cv2.bitwise_not(mask)

img1_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
img2_fg = cv2.bitwise_and(img1,img1,mask=mask)
total = cv2.add(img1_bg,img2_fg)
cv2.imshow("image",img1_bg)
cv2.imshow("image2",img2_fg)
cv2.imshow("image3",total) 
'''
 arkaplan 2.resim oldu, 1.resmi üzerine eklemis oldu. Roi islemi yapilmasaydı cok uzun bir carpma islemi yapılacaktı. 
 Ve cok fazla islem gucu gerekecekti.

'''

cv2.waitKey(0)
cv2.destroyAllWindows()
