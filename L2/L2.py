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

    cv2.imshow('Gradiente en x sobel operator', grad_x_scaled)
    cv2.waitKey(0)
    cv2.imshow('Gradiente en y sobel operator', grad_y_scaled)
    cv2.waitKey(0)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    mod = np.sqrt(np.uint32(np.power(grad_x,2) + np.power(grad_y,2)))

    cv2.imshow('Modulo Sobel operator', np.uint8(mod))
    cv2.waitKey(0)
    # sobel_X = cv2.Sobel(imagen, cv2.CV_64F, 1, 0) 
    # sobel_X_abs = np.uint8(np.absolute(sobel_X)) 
    # sobel_Y = cv2.Sobel(imagen, cv2.CV_64F,0, 1) 
    # sobel_Y_abs = np.uint8(np.absolute(sobel_Y)) 

    # sobel_XY_combined = cv2.bitwise_or(sobel_Y_abs,sobel_X_abs)
    orientacion = np.arctan2(grad_y,grad_x)
    cv2.imshow('Direction Sobel operator', np.uint8(orientacion))
    cv2.waitKey(0)


def gaussiana(sigma):
    n = 5 * sigma if sigma % 2 ==1 else 5 * sigma + 1
    G = np.zeros(n, float)
    inf_limit = int(-((n -1) /2))
    sup_limit = int((n-1)/ 2+1)
    i = 0
    for x in range(inf_limit,sup_limit, 1):
       G[i] = math.exp(-((x **2)/ (2*(sigma **2))))
       i = i +1
    return G

def gaussianaDerivada(sigma):
    n = 5 * sigma if sigma % 2 ==1 else 5 * sigma + 1
    G = np.zeros(n, float)
    inf_limit = int(-((n -1) /2))
    sup_limit = int((n-1)/ 2+1)
    i = 0
    for x in range(inf_limit,sup_limit, 1):
       G[i] = - x / (sigma ** 2) * math.exp((- (x ** 2)) / (2 * (sigma **2)))
       i = i +1
    return G



def K(G, G2):
    K1 = 0
    K2 = 0
    for valor in G:
        if valor > 0:
            K1 += valor 
    for valor in G2:
        if valor > 0:
            K2 += valor
    return K1 * K2

def gaussian(x : float, sigma : float):
    return np.exp((-x ** 2) / (2 * (sigma **2)))

def gaussianDerivative(x : float, sigma : float):
    return -x / (sigma ** 2) * np.exp((-x ** 2) / (2 * (sigma ** 2)))

def cannyOperator(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sigma = 1
    
    G = gaussiana(sigma)[:, np.newaxis]
    G2 = gaussianaDerivada(sigma)[np.newaxis, :]

    G2_inv = -G2
   
    convx1 = cv2.filter2D(gray, cv2.CV_64F, cv2.flip(G,-1), borderType=cv2.BORDER_CONSTANT)
    grad_x = cv2.filter2D(convx1, cv2.CV_64F, cv2.flip(G2_inv,-1), borderType=cv2.BORDER_CONSTANT)
    
    grad_x_scaled = np.uint8(grad_x /2 + 128)
    
    cv2.imshow('Gradiente en x canny operator', grad_x_scaled)
    cv2.waitKey(0)

    G = gaussiana(sigma)[np.newaxis, :]
    G2 = gaussianaDerivada(sigma)[:, np.newaxis]
    convy1 = cv2.filter2D(gray,cv2.CV_64F, cv2.flip(G2,-1), borderType=cv2.BORDER_CONSTANT)
    grad_y = cv2.filter2D(convy1,cv2.CV_64F, cv2.flip(G,-1), borderType=cv2.BORDER_CONSTANT)
    


    
    grad_y_scaled = np.uint8(grad_y / 2 +128)

    
    cv2.imshow('Gradiente en y canny operator', grad_y_scaled)
    cv2.waitKey(0)
    
 #   mod = np.sqrt(np.uint32(np.power(grad_x,2) + np.power(grad_y,2)))

  #  cv2.imshow('Modulo canny operator', np.uint8(mod))
   # cv2.waitKey(0)

    # orientacion = np.arctan2(grad_y,grad_x)
    # cv2.imshow('Direction canny operator', np.uint8(orientacion))
    # cv2.waitKey(0)



cap = cv2.VideoCapture(0)
img = cv2.imread('poster.pgm', cv2.IMREAD_COLOR)

SobelOperator(img)
cannyOperator(img)

cv2.destroyAllWindows()