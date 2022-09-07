# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 12:15:03 2022

@author: Elif
"""
import cv2

cam = cv2.VideoCapture(0)

'''
kameranın genisligi, uzunlugu, fps gibi degerleriin goruntulenmesi
'''
# print(cam.get(3)) # kameranızın genisligi
# print(cam.get(4)) # kameranızın uzunlugu
print(cam.get(5))   # kameranızın calistigi fps sayısı

'''
boyutlarının yeniden ayarlanması, set metoduyla olur

'''

cam.set(3,320) # guncellenecek degerler
cam.set(4,240) 

# guncellenmis sonuclar
print(cam.get(3)) # kameranızın genisligi
print(cam.get(4)) # kameranızın uzunlugu

if not cam.isOpened():
    print("camera is not recognized")
    exit()
    
while True:
    ret, frame = cam.read()
    
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    if not ret:
        print("okunamiyor")
        break
    
    cv2.imshow("kamera",frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("goruntu sonlandirildi")
        break
    
cam.release()