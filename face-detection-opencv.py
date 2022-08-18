# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 21:05:29 2022

@author: Elif
"""
import cv2

detection_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(0)

while(True): # kameradan surekli goruntu okumak ıcın sonsuz dongu kurulur.
    
    info_read, color_image_info = camera.read() # read() fonksiyonu ile kameradan anlık goruntu okunur.
    
    """
    Okunan goruntu bilgisi color_image_info degiskenine aktarilir.
    Sorunsuz okunursa info_read True olarak donecektir.
    
    """
    
    """
    Haar Cascade modeli yalnızca gri seviyeli görüntülerde yüz tespiti 
    yapmaktadır. read() fonksiyonu ile okuduğumuz görüntü renkli bir görüntü
    olduğundan öncelikle cvtColor(, cv2.COLOR_BGR2GRAY) fonksiyonu ile 
    gri seviyeye çevrilmektedir.
   
    """
   
    gri_image = cv2.cvtColor(color_image_info, cv2.COLOR_BGR2GRAY)
    
    """
    detectMultiScale(GRI_GORUNTU, 1.1, 4) fonksiyonu ile gri görüntü
    içerisindeki yüz görüntüleri bulunmaktadır.
    
    """
    detected_face = detection_model.detectMultiScale(gri_image,1.1,4)
    
    for(x,y,w,h) in detected_face:
        
        """
        
        Koordinatlar gri seviyeli görüntü üzerinde bulunduğu anda gri seviyeli
        görüntüyü tekrar kullanmayız. Onun yerine bulunan yüz etrafına bir
        dikdörtgen çizmek için okunan renkli görüntüyü kullanırız.
        cv2.rectangle() fonksiyonu verilen koordinatlara göre dikdörtgen
        oluşturacaktır.
    
        """
        
        cv2.rectangle(color_image_info, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    cv2.imshow('Face Detection Application', color_image_info)
    
    if cv2.waitKey(30) & 0xff == 27:
        
        """
        
        Burada 27 ESC'ye karşılık gelmektedir. Programı durdurmak için ESC tuşuna
        basmalıyız.
    
        """
        
        cv2.destroyAllWindows()
        
        break
        """
    
         Tüm işlemler bittikten sonra kameranın kapanması için bu komut eklenir.
     
        """
camera.release()
