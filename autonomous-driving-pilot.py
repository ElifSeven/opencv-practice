# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 21:43:23 2022

@author: Elif
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np


img_path = "/roadd.jpg"
img = cv2.imread(img_path)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(thresh,output2) = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

output2 = cv2.GaussianBlur(output2, (5, 5), 3)
output2 = cv2.Canny(output2, 180, 255)
plt.imshow(output2)
plt.show()


# without blurring
'''
bulanıklastırma islemi olmazsa, kirlilik artar. Eğer olmazsa sonucunda gurultuler olusur.
'''

'''
Bu islemlerden sonra Canny Edge Detection islemi:
    1.parametre. resim
    2.parametre: low threshold value
    3.parametre: high threshold value

'''

img = cv2.imread(img_path)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(thresh, output2) = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
output2 = cv2.Canny(output2, 180, 255)
plt.imshow(output2)
plt.show()



'''
Bu aşamadan sonra belirlenen kenarlar esas alınarak gerçek resim üzerine işlemler yapılır.
 Bunun için HoughLinesP ve line fonksiyonları kullanılır.
'''

lines = cv2.HoughLinesP(output2, 1, np.pi/180,30)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),4)
plt.imshow(img)


'''
 yol sınırları ve şeritler güzel bir şekilde elde edildi. 
 Ancak resim dikkatli bir şekilde incelendiğinde,
 Şerit ve yol sınırlarını tespit etmesinde problem olmamasına rağmen
 agaclar da sanki yol sınırıymış gibi algılanmıştır. Çözüm: maskeleme
'''

def mask_of_image(image):
    height = image.shape[0]
    polygons = np.array([[(0,height),(2200,height),(250,100)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

