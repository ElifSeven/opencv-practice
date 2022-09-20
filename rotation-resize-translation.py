# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 17:12:02 2022

@author: Elif
"""

import cv2
import numpy as np

img = cv2.imread("1.jpeg")

print(img.shape)

'''
 yeniden boyutlandırma
'''
# img2 = cv2.resize(img,(300,300))

# img2 = cv2.resize(img,None,fx = 1.5,fy=1.5)

# img2 = cv2.resize(img,None,fx = 0.5,fy=0.5, interpolation = cv2.INTER_CUBIC)

''' 
Translation (Goruntu uzerinde yer degistirme)

'''

translation_matrix = np.float32([
    [1,0,50], # x tarafından ne kadar gidecegi bilgisi. Ters gitsin dersek - yazmak yeterli.
    [0,1,50]]) # y tarafından ne kadar gidecegi bilgisi
''' Ana resim boyuytları aynı kalarak 50 px asagı 50 px yukarı gitti.
'''

rows,cols = img.shape[:2] # 3 boyutlu oldugu icin ilk 2 parametreyi aldık

# 3.parametre yer degistirmeden sonraki elde edilen boyutlardır. Orijinal goruntunun boyutları olarak alabiliriz.
img_translation = cv2.warpAffine(img,translation_matrix,(cols,rows))
''' Kırpma islemi olmadan yerlestirilmesi icin +50 ekledik
'''

img_translation = cv2.warpAffine(img,translation_matrix,(cols+50,rows+50))


''' 
Rotation Islemi

'''
rotation_matrix = cv2.getRotationMatrix2D((cols/2,rows/2),-50,1)

img_rotation = cv2.warpAffine(img,rotation_matrix,(cols,rows))

cv2.imshow("img",img_rotation )

cv2.waitKey()
cv2.destroyAllWindows()