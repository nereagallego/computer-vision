# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 15:00:50 2023

@author: nerea y victor
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from operator import itemgetter

def SobelOperator(imagen):

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    imagen = cv2.GaussianBlur(imagen, (3, 3), 0)

    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
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
    
    k = K(gaussiana(sigma),gaussianaDerivada(sigma))
   
    convx1 = cv2.filter2D(gray, cv2.CV_64F, cv2.flip(G,-1), borderType=cv2.BORDER_CONSTANT)
    grad_x = cv2.filter2D(convx1, cv2.CV_64F, cv2.flip(G2_inv,-1), borderType=cv2.BORDER_CONSTANT)
    
    grad_x_scaled = np.uint8(grad_x /2 + 128)
    
    # cv2.imshow('Gradiente en x canny operator', grad_x_scaled)
    # cv2.waitKey(0)

    G = gaussiana(sigma)[np.newaxis, :]
    G2 = gaussianaDerivada(sigma)[:, np.newaxis]
    convy1 = cv2.filter2D(gray,cv2.CV_64F, cv2.flip(G2,-1), borderType=cv2.BORDER_CONSTANT)
    grad_y = cv2.filter2D(convy1,cv2.CV_64F, cv2.flip(G,-1), borderType=cv2.BORDER_CONSTANT)
    


    
    grad_y_scaled = np.uint8(grad_y / 2 +128)

    
    # cv2.imshow('Gradiente en y canny operator', grad_y_scaled)
    # cv2.waitKey(0)
    
    mod = np.sqrt(np.uint32(np.power(grad_x,2) + np.power(grad_y,2)))

    # cv2.imshow('Modulo canny operator', np.uint8(mod))
    # cv2.waitKey(0)

    orientacion = np.arctan2(grad_y,grad_x)
    # cv2.imshow('Direction canny operator', np.uint8(orientacion))
    # cv2.waitKey(0)
    return grad_x, grad_y, mod, orientacion

def vanishPointing(img, gx, gy, mod, orientation):
    size_x, size_y = img.shape[1], img.shape[0]
    index_fila_central = int(size_y/2)


    accumulator = np.zeros(size_x, dtype=np.uint64)

    for i in range(size_y):
        for j in range(size_x):
            if j == 498 and i == 9:
                print()
            if mod[i,j] > 131:
                x = j - size_x / 2
                y = size_y / 2 - i
                theta = orientation[i,j]
                if not (abs(theta- 0) < 0.01 or abs(theta - math.pi) < 0.01 or abs(theta + math.pi) < 0.01 or abs(theta - math.pi/2) < 0.01 or abs(theta + math.pi/2) < 0.01):
                    p = x * math.cos(theta) + y * math.sin(theta)
                    x_int = p / math.cos(theta)

                    # index en img
                    k = round(x_int + size_x / 2)
                    if k >= 0 and k < size_x:
                        accumulator[k] +=1
    # segundo punto más votado
    numbers_sort = sorted(enumerate(accumulator), key=itemgetter(1),  reverse=True)
    index, value = numbers_sort[1]
    return index_fila_central, np.argmax(accumulator)
    
    
    
    
img = cv2.imread('pasillo2.pgm', cv2.IMREAD_COLOR)

# SobelOperator(img)
gx, gy, mod, orientation = cannyOperator(img)
# grad_x_scaled = np.uint8(gx /2 + 128)
# grad_y_scaled = np.uint8(gy / 2 +128)
# cv2.imshow('Gradiente en x canny operator', grad_x_scaled)
# cv2.waitKey(0)
# cv2.imshow('Gradiente en y canny operator', grad_y_scaled)
# cv2.waitKey(0)
# cv2.imshow('Modulo canny operator', np.uint8(mod))
# cv2.waitKey(0)
# cv2.imshow('Direction canny operator', np.uint8(orientation))
# cv2.waitKey(0)
y1, x1 = vanishPointing(img, gx, gy, mod, orientation)

cv2.putText(img,'+', (x1,y1), 0, 1, (255,0,0),2)
# cv2.putText(img,'+', (x2,y2), 0, 1, (0,0,255),2)
cv2.imshow('imagen',img)
cv2.waitKey(0)
cv2.destroyAllWindows()