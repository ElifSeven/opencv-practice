# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:24:33 2022

@author: Elif
"""
'''
1. Yontem : Affine Transformations
'''
import cv2
import numpy as np

img = cv2.imread("1.jpeg")

print(img.shape)

rows,cols = img.shape[0:2]

# 3 nokta secerek olusturuldugu icin bu yontemde 3 deger alıyor. 2 tane matris olusturucaz.

'''src_point = np.float32([
    [0,0],
    [cols-1,0],
    [0,rows-1]])

# yeni olsucak resimde hangi noktalara hangi piksele denk gelecegi
dst_points = np.float32([
    [0,0],
    [int(0.6*(cols-1)),0], # yukarıdakinin aynı noktayı %60 sola cekmek icin
    [int(0.4*(cols-1)),rows-1]])

# transformation
affine_matrix = cv2.getAffineTransform(src_point,dst_points)

# sonuc matrisini resim ile carpma islemi:
img_output = cv2.warpAffine(img,affine_matrix,(cols,rows))
 



cv2.imshow("img",img )
cv2.imshow("img_output",img_output )

cv2.waitKey()
cv2.destroyAllWindows()
'''

'''
2. Yontem : Projective Transformations
'''
# bu yontemde 4 nokta seciliyor sonra yeni resimde o 4 noktanın nereye gelecegini belirliyoruz.

src_point = np.float32([
    [0,0], # sol üst
    [cols-1,0], # sag ust
    [0,rows-1], # sol alt
    [cols-1,rows-1]]) # sag alt

dst_point = np.float32([
    [0,0], # sol üst
    [cols-1,0], # sag ust
    [int(0.33*(cols-1)),rows-1], # sol alt
    [int(0.66*(cols-1)),rows-1]]) # sag al

projective_matrix = cv2.getPerspectiveTransform(src_point,dst_point)

img_output = cv2.warpPerspective(img,projective_matrix,(cols,rows))

cv2.imshow("img",img )
cv2.imshow("img_output",img_output )

cv2.waitKey()
cv2.destroyAllWindows()