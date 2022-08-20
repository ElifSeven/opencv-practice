# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 16:52:49 2022

@author: Elif
"""
import cv2
from matplotlib import pyplot as plt



img_path = "/road.jpg"
img = cv2.imread(img_path)
print(img.shape)

# to gray-scale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# to display new image
plt.imshow(gray_image)
plt.show()
print(gray_image.shape)

'''
matrisin yalnızca 0 ve 255 değerlerinden oluşmasını isteyebiliriz.
Böyle durumlarda threshold fonksiyonu kullanılmaktadır.
2.parametre belirnen eşik degeri, bu esik degerini gecen matris elemanlarının atanmasını istedigimiz deger.
'''
(thresh, blackAndWhiteImage) = cv2.threshold(gray_image, 20, 255, cv2.THRESH_BINARY)
(thresh, blackAndWhiteImage) = cv2.threshold(gray_image, 80, 255, cv2.THRESH_BINARY)
(thresh, blackAndWhiteImage) = cv2.threshold(gray_image, 160, 255, cv2.THRESH_BINARY)
(thresh, blackAndWhiteImage) = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
plt.imshow(blackAndWhiteImage)
plt.show()


# blur image
blurred_output = cv2.blur(gray_image,(10,10))
plt.imshow(blurred_output)
plt.show()

# gaussian blur
blurred_gaussian = cv2.GaussianBlur(gray_image,(9,9),5)
plt.imshow(blurred_gaussian)
plt.show()


# rotation image
'''

resmin merkezi belirlenerek döndürme işlemi  bu merkeze göre gerçekleştirilir.
3 parametre alır: 
    1.parametre : hesaplanan merkez değerleri
    2.parametre: açı değeri
    3.parametre: dondurme işlemi sonucunda olcekleme degeri. Bu deger buyutukduce
    resimde merkeze gore buyutme gerceklesir.
    .
'''
(h, w) = img.shape[:2]
center = (w / 2, h / 2)
M = cv2.getRotationMatrix2D(center, 13, scale  =1.5)
rotated = cv2.warpAffine(gray_image, M, (w, h))
plt.imshow(rotated)
plt.show()