# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 15:25:08 2018

@author: Aneesh
"""
import tensorflow as tf
import numpy as np
import cv2
import math
from scipy import ndimage
#import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)
    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx,shifty
def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted  
def cap_img(n):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter = 0
    gray =[]
    images = np.zeros((n,784))
    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)
    
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            gray = cv2.imread(img_name,0)
            gray = cv2.resize(255-gray, (28, 28))
            (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            while np.sum(gray[0]) == 0:
                gray = gray[1:]
            while np.sum(gray[:,0]) == 0:
                gray = np.delete(gray,0,1)
            while np.sum(gray[-1]) == 0:
                gray = gray[:-1]    
            while np.sum(gray[:,-1]) == 0:
                gray = np.delete(gray,-1,1)
            rows,cols = gray.shape
            if rows > cols:
              factor = 20.0/rows
              rows = 20
              cols = int(round(cols*factor))
              gray = cv2.resize(gray, (cols,rows))
            else:
              factor = 20.0/cols
              cols = 20
              rows = int(round(rows*factor))
              gray = cv2.resize(gray, (cols, rows))
            colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
            rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
            gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
            shiftx,shifty = getBestShift(gray)
            shifted = shift(gray,shiftx,shifty)
            gray = shifted
            img_counter += 1
            flatten = gray.flatten() / 255.0
            images[img_counter] = flatten
            print("{} written!".format(img_name))
    return(images)
    
def reshapm(x_train,x_test):
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return(x_train,x_test)
def reshapt(x_train):
    x_train = np.reshape(x_train,(x_train.shape[0], 28, 28, 1))
    x_train = x_train.astype('float32')
    x_train /= 255
    return(x_train)
def main():
    #let us train the model first
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    (x_train, x_test) = reshapm(x_train,x_train)
    input_shape = (28,28,1)
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(x=x_train,y=y_train, epochs=3)
    print("please enter the number of pictures you want to classify")
    n = int(input())
    # now let us capturee the images
    image = cap_img(n)
    image = reshapt(image)
    image = np.array(image)
    pred = model.predict_classes(image, verbose=2)
    print("Predicted=%s" % (pred))

main()
    