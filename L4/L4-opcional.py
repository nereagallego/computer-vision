# -*- coding: utf-8 -*-
"""
Created on Wed May  3 20:27:04 2023

@author: nerea
"""

import numpy as np
import cv2
import tarfile
from matplotlib import pyplot as plt
import random
import math

import time
import os

"""_summary_ Devuelve los keypoints, los descriptores, el tiempo de detección y la cantidad de características

Args:
    nfeatures: number of features returned
    contrstThreshold: filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
    sigma: The sigma of the Gaussian applied to the input image at the octave #0

"""
def SIFT_keypoints(gray, nfeatures : int, contrastThreshold=0.04, sigma=1.6):
    start = time.time()
    sift = cv2.SIFT_create(nfeatures)
    kp, desc = sift.detectAndCompute(gray,None)
    end = time.time()
    return kp, desc, end - start, len(kp)

# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
def flann_Matching(img1,kp1, desc1,img2, kp2, desc2):
    # FLANN parameters
    start = time.time()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(desc1,desc2,k=2)
    end = time.time()
    time_emparejamiento = end - start
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    good = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            
            good.append(matches[i])
    good = []
    i = 0
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
            good.append(m)
            i += 1
    # draw_params = dict(matchColor = (0,255,0),
                    # singlePointColor = (255,0,0),
                    # matchesMask = matchesMask,
                    # flags = cv2.DrawMatchesFlags_DEFAULT)
    # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    # plt.imshow(img3,),plt.show()
            
    return time_emparejamiento, len(matches), good


def calculate_RANSAC_function(gray, gray2):
    añadir = False
    kp1, desc1, time_detection, num_features = SIFT_keypoints(gray,1000)
    kp2, desc2, time_detection2, num_features2  = SIFT_keypoints(gray2,1000)
    time1, num_matches1, matches1 = flann_Matching(gray, kp1, desc1, gray2, kp2, desc2)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches1 ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches1 ]).reshape(-1,1,2)

    if len(matches1) < 4:
        return None, False
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    if M is None:
        return None, False
    matchesMask = mask.ravel().tolist()

    h,w = gray.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    
    dst = cv2.perspectiveTransform(pts,M)
    

    list_of_points = np.concatenate(dst, axis=0)
    
    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel())
    
    if abs(x_min) < 150 and  abs(x_min) > 20:
        añadir = True
    # img2 = cv2.polylines(gray2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       # singlePointColor = None,
                       # matchesMask = matchesMask, # draw only inliers
                       # flags = 2)
    # img3 = cv2.drawMatches(gray,kp1,gray2,kp2,matches1,None,**draw_params)
    # plt.imshow(img3, 'gray')
    # plt.show()
    return M, añadir

def trim(img,tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]


def construct_panorama(gray, gray2, H):
    
    h,w = gray.shape
    h2, w2 = gray2.shape
    pts = np.float32([ [0,0],[0,h],[w,h],[w,0] ]).reshape(-1,1,2)
    pts2 = np.float32([[0,0], [0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,H)
    # dst = dst.reshape(4,2)

    list_of_points = np.concatenate((pts2, dst), axis=0)
    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel())
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel())
        
    translation_dist = [-x_min,-y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])
    output_img = cv2.warpPerspective(gray, 
                                     H_translation.dot(H), 
                                     (x_max - x_min, y_max - y_min))

    output_img[translation_dist[1]:h2+translation_dist[1], 
               translation_dist[0]:w2+translation_dist[0]] = gray2
    plt.imshow(output_img)
    plt.show()
    
    return trim(output_img)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
base = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
plt.imshow(base)
plt.show()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    H, añadir = calculate_RANSAC_function(gray2, base)
    if añadir:
        base = construct_panorama(gray2, base, H)

    if cv2.waitKey(1) == ord('q'):
        break