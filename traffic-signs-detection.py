# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 19:39:20 2022

@author: Elif
"""

import pickle
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Model
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator


from tensorflow.keras.optimizers import Adam # - Works

num_classes = 43


def get_data(train_path,val_path,test_path):
    with open(train_path,"rb") as f:
        train_data = pickle.load(f)
    with open(val_path,"rb") as f:
        val_data = pickle.load(f)
        
    with open(test_path,"rb") as f:
        test_data = pickle.load(f)
        
        
    X_train, y_train = train_data["features"], train_data["labels"]
    X_val, y_val = val_data["features"],val_data["labels"]
    X_test, y_test = test_data["features"],test_data["labels"]
    
    
    return X_train, y_train, X_val, y_val, X_test, y_test

train_path = "train.p"
val_path = "valid.p"
test_path = "test.p"

X_train,y_train,X_val,y_val,X_test, y_test = get_data(train_path, val_path, test_path)


plt.imshow(X_train[3344])
print(X_train.shape)

print(y_train[3344])

sign_names = pd.read_csv("signnames.csv")
print(sign_names)




'''
Verilerin yapılarının anlaşılmasının ardında bu yapılar siyah-beyaz hale getirilir.
 Burada unutulmaması gereken unsur bu işlem her veri setinde yapılamaz.
 Renklerin önem arz ettiği durumlarda resimler renkli şekilde işlenmelidir. 
 Ancak bu projede resimlerin renkli olmalarının eğitim üzerinde herhangi bir etkisi bulunmamaktadır.
'''
def image_process(img):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # siyah-beyaz formata cevirilmesi
    equalized_img = cv2.equalizeHist(gray_img) # resmin histogram degerlerinin esitlenmesi
    final_image = equalized_img/255 # elde edilen matris degerlerine normalizasyon
    return final_image

X_train = np.array(list(map(image_process,X_train)))
X_test = np.array(list(map(image_process,X_test)))
X_val = np.array(list(map(image_process,X_val)))

plt.imshow(X_train[3344])
print(X_train.shape)



'''
Bu resimlerin eğitime girebilmesi için boyutlarının ayarlanması gerekir. 
Oluşturacağımız model her bir resim için 3 boyutlu bir matrise ihtiyaç duymaktadır. 
Ayrıca etiket değerlerine içeren y_train, y_test ve y_val setleri de kategorik vektör olacak şekilde düzenlenir 
(One Hot Encoding olarak yapılandırılır).
'''
# reshaping
X_train = X_train.reshape(34799,32,32,1)
X_test = X_test.reshape(12630,32,32,1)
X_val = X_val.reshape(4410,32,32,1)
                   
# One Hot Encoding
y_train = to_categorical(y_train,43)
y_test = to_categorical(y_test,43)
y_val = to_categorical(y_val,43)

# Model Tasarım
def model_1():
    model = Sequential()
    model.add(Conv2D(60,(5,5), input_shape = (32,32,1), activation = "relu"))
    model.add(Conv2D(60, (5,5),  activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(30, (3,3), activation = "relu"))
    model.add(Conv2D(30, (3,3), activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))
    #model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation ="softmax"))
    model.compile(Adam(lr= 0.01), loss = "categorical_crossentropy", metrics =[ "accuracy"])
    return model
model = model_1()
print(model.summary())

history = model.fit(X_train, y_train, epochs = 10, validation_data = (X_val,y_val), batch_size = 400, verbose = 1, shuffle = 1)

# augmentation teknikleri 
'''
Veri artırım işlemi için ImageDataGenerator kullanılmıştır.
 Buradaki width_shift_range ile 0.1 oranında resim genişliği kayar, 
 height_shift_range ile 0.1 oranında resim yükseliği kayar, 
 zoom_range ile 0.1 oranında resim üzerine zoom yapılır, 
 shear_range ile resim üzerinde eğiklik meydana getirilir.
 Son olarak da rotation_range ile resim 10 derece açı oluşturacak şekilde döndürülür.
'''
def data_aug():
    data_generator= ImageDataGenerator(height_shift_range = 0.1,width_shift_range = 0.1, shear_range=0.1, zoom_range =0.1,rotation_range=10)
    data_generator.fit(X_train)
    return data_generator

# after some features
def model_2():
    model = Sequential()
    model.add(Conv2D(80, (5,5), input_shape = (32,32,1), activation = "relu"))
    model.add(Conv2D(80, (5,5),  activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(40, (3,3), activation = "relu"))
    model.add(Conv2D(40, (3,3), activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))
    #model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation ="softmax"))
    model.compile(Adam(lr= 0.001), loss = "categorical_crossentropy", metrics =[ "accuracy"])
    return model
model = model_2()
print(model.summary())
