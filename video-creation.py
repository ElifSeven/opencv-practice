# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 21:12:00 2022

@author: Elif
"""

# video olusturma

import cv2

cam = cv2.VideoCapture(0)

fourrc = cv2.VideoWriter_fourcc("X", "V", "I", "D")

out = cv2.VideoWriter("created_video.avi",fourrc,30.0,(640,480))

while cam.isOpened():
    
    ret, frame = cam.read()
    
    if not ret:
        print("kameradan goruntu alınamadı")
        break
    
    out.write(frame)
    cv2.imshow("kamera",frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("videodan ayrıldınız")
        break
    
cam.release()
out.release()
cv2.destroyAllWindows()