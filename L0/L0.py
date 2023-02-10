# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 13:21:59 2023

@author: nerea
"""

# #%% Example 1: read and show an image
# import cv2 # Import python-supported OpenCV functions
# import numpy as np # Import numpy and call it np
# from matplotlib import pyplot as plt # Import pyplot and call it plt
# img = cv2.imread('starry_night.jpg', cv2.IMREAD_GRAYSCALE)
# cv2.namedWindow( 'Example1', cv2.WINDOW_AUTOSIZE )
# cv2.imshow('Example1',img)
# print(type(img))
# print(img.shape)
# cv2.waitKey(0)
# cv2.destroyWindow( 'Example1' ) # cv2.destroyAllWindows()

#%% Matrices in OpenCV: properties
import cv2 # Import python-supported OpenCV functions
import numpy as np # Import numpy and call it np
from matplotlib import pyplot as plt # Import pyplot and call it plt
def show_image_properties(my_img):
    cv2.namedWindow( 'Example1', cv2.WINDOW_AUTOSIZE )
    cv2.imshow('Example1',my_img)
    print("Properties of the matrix used to store the image")
    print("They are numpy arrays: type(my_img)= ", type(my_img))
    print("Rows, columns and channels: my_img.shape= ", my_img.shape)
    print("Total number of pixels: my_img.size= ", my_img.size)
    print("Image datatype: my_img.dtype = ", my_img.dtype )
    cv2.waitKey(1000) #cv2.waitKey(0)
    cv2.destroyWindow( 'Example1' ) # cv2.destroyAllWindows()
    return(0)
img = cv2.imread('starry_night.jpg', cv2.IMREAD_GRAYSCALE)
img_bgr = cv2.imread('starry_night.jpg', cv2.IMREAD_COLOR)
show_image_properties(img)
show_image_properties(img_bgr)

#%% Matrices in OpenCV: elements
# Create a black image (BGR: uint8, 3 channels)
img_uint8 = np.zeros((512,512,3), np.uint8)
# Draw a diagonal BLUE line with thickness of 5 px
cv2.line(img_uint8,(0,0),(511,511),(255,0,0),5)
show_image_properties(img_uint8)
# Create a black image (BGR: uint8, 3 channels)
img_uint8b = np.zeros((512,512,3), np.uint8)
# Access to elements and set them the value BLUE
img_uint8b[0:100,0:25]=[250,0,0]
# Alternatively..
#img_uint8b[0:100,0:25,0]=250
#img_uint8b[0:100,0:25]=np.array([250,0,0], np.uint8)
show_image_properties(img_uint8b)
# Try and observe the following mistakes (common mistakes; note that often there is not even a warning..)
img_uint8d = np.zeros((512,512,3), np.uint8)
img_uint8d[0:100,0:25,0]= -100 #in some versions, no warning!!
img_uint8d[0:100,0:25,0]= -500 #in some versions, no warning!!
show_image_properties(img_uint8d)

#%% Operations with matrices
# m1 = np.matrix([[1, 2],[3, 4]])
# m2 = m1 # new reference to m1, data is not copied
# m3 = m1.copy() # complete copy of m1
# m3 += 5.0
# m3 = 0.5 * m1 + 0.3 * m2
# A = np.eye(3, dtype = np.float32)
# X = np.eye(3, dtype = np.float32)
# Y = A * X
# Y = np.dot(A, X) # observe the difference
# Y = A.T * X # transpose
# X = np.linalg.inv(A) * Y; # inverse
# cv2.solve(A, Y ,X); # solves the system A*X = Y
# v3 = np.cross(v1, v2) # vector product
# z = np.dot(v1, v2) # scalar product
# m3 = m1 > m2 # returns a matrix of numpy.bool_
# dst = cv2.add(img1, img2) # sums all channels
# masked_img = cv2.bitwise_and(img,img,mask = mask) # mask

#%% How to traverse a matrix
import random as r
def salt(image, n):
# Adds salt noise to n pixels randomly selected in the image
    for k in range(n):
        i= r.randrange(image.shape[0]) # range of image rows
        j= r.randrange(image.shape[1]) # range of image cols
        if len(image.shape) == 2: # Gray image
            image[i,j]= 255
        else: # RGB image
            image[i,j,0]= 255
            image[i,j,1]= 255
            image[i,j,2]= 255
    return image 

img = cv2.imread('starry_night.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('starry_night.jpg', cv2.IMREAD_COLOR)
img2=salt(img, 10000)
cv2.imshow('Original', img)
cv2.imshow('Salt', img2)
cv2.waitKey(0)
cv2.destroyAllWindows

import cv2 # Import python-supported OpenCV functions
import numpy as np # Import numpy and call it np
def colorReduce1(img_in, div=np.uint8(32)):
    rows,cols = img_in.shape # grayscale images
    print(rows,cols)
    img2 = img_in.copy()
    for i in range(rows):
        for j in range(cols):
            img2[i,j] = np.uint8(np.uint8(img_in[i,j] /div)*div)
    return(img2)
img_bgr = cv2.imread('starry_night.jpg', cv2.IMREAD_GRAYSCALE)
img2 = colorReduce1(img_bgr)
cv2.imshow('A', img_bgr)
cv2.imshow('Reduced',img2)
cv2.waitKey()
cv2.destroyAllWindows()

#%% Iterating Arrays Using ndenumerate()
img = np.array([[1, 2, 3], [4, 5, 6]])
print("Initial matrix:")
print(img)
img2 = img.copy()
for idx, x in np.ndenumerate(img):
    print("idx=", idx)
    print("x=", x)
    
#%%
import cv2 # Import python-supported OpenCV functions
import numpy as np # Import numpy and call it np
def colorReduce2(img_in, div=np.uint8(64)):
    img2 = img_in.copy()
    for idx, x in np.ndenumerate(img_in): # one or three channels
        img2[idx] = np.uint8(np.uint8( x /div)*div)
    return(img2)
#img_bgr = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
img_bgr = cv2.imread('starry_night.jpg', cv2.IMREAD_COLOR)
img2 = colorReduce2(img_bgr)
cv2.imshow('A', img_bgr)
cv2.imshow('Reduced',img2)
cv2.waitKey()
cv2.destroyAllWindows()

num_runs = 10
e1 = cv2.getTickCount()
for k in range(num_runs):
    img2 = colorReduce1(img_bgr)
e2 = cv2.getTickCount()
for k in range(num_runs):
    img3 = colorReduce2(img_bgr)
e3 = cv2.getTickCount()
t1 = (e2 - e1)/cv2.getTickFrequency()
t2 = (e3 - e2)/cv2.getTickFrequency()
print( num_runs, " executions of color reduce. Time t1=", t1,
"seconds and time t2=", t2, "seconds." )

#%% tomar una imagen de la cámara
import cv2
cap = cv2.VideoCapture(0)

leido, frame = cap.read()

if leido == True:
	#cv2.imwrite("foto.png", frame)
    cv2.imshow('Foto tomada', frame)
    cv2.waitKey()
    cv2.destroyAllWindows()
    print("Foto tomada correctamente")
else:
	print("Error al acceder a la cámara")