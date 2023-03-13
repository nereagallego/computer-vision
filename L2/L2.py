# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 15:00:50 2023

@author: nerea y victor
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

def SobelOperator(imagen):

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    imagen = cv2.GaussianBlur(imagen, (3, 3), 0)

    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    

    grad_x_scaled = np.uint8(grad_x /2 + 128)
    grad_y_scaled = np.uint8(grad_y / 2 +128)

    cv2.imshow('Gradiente en x', grad_x_scaled)
    cv2.waitKey(0)
    cv2.imshow('Gradiente en y', grad_y_scaled)
    cv2.waitKey(0)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    mod = np.sqrt(np.power(grad_x,2) + np.power(grad_y,2))

    cv2.imshow('Sobel operator', np.uint8(mod))
    cv2.waitKey(0)
    # sobel_X = cv2.Sobel(imagen, cv2.CV_64F, 1, 0) 
    # sobel_X_abs = np.uint8(np.absolute(sobel_X)) 
    # sobel_Y = cv2.Sobel(imagen, cv2.CV_64F,0, 1) 
    # sobel_Y_abs = np.uint8(np.absolute(sobel_Y)) 

    # sobel_XY_combined = cv2.bitwise_or(sobel_Y_abs,sobel_X_abs)
    orientacion = np.arctan2(grad_y,grad_x)
    cv2.imshow('Sobel operator', np.uint8(orientacion))
    cv2.waitKey(0)


 #def cannyOperator(img):
    


cap = cv2.VideoCapture(0)
img = cv2.imread('poster.pgm', cv2.IMREAD_COLOR)

SobelOperator(img)
cv2.waitKey(0)
cv2.destroyAllWindows()