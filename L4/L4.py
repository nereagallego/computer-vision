import numpy as np
import cv2
import tarfile
from matplotlib import pyplot as plt

import time

# https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
"""
    Devuelve los keypoints, los descriptores, el tiempo de detección y la cantidad de características
"""
def SIFT_keypoints(gray, nfeatures):
    start = time.time()
    sift = cv2.SIFT_create(nfeatures)
    kp, desc = sift.detectAndCompute(gray,None)
    end = time.time()
    return kp, desc, end - start, len(kp)
    

# https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
def HARRIS_keypoints(gray):
    
    dst = cv2.cornerHarris(gray,2,3,0.04)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    
    return dst

# https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html
def ORB_keypoints(gray, nfeatures):

    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures)
    # find the keypoints with ORB
    kp = orb.detect(gray,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(gray, kp)
    # draw only keypoints location,not size and orientation
    return kp, des

def AKAZE_keypoints(gray):
    akaze = cv2.AKAZE_create()
    kpts1, desc1 = akaze.detectAndCompute(gray, None)
    return kpts1

# https://docs.opencv.org/3.4/d3/da1/classcv_1_1BFMatcher.html
"""
    L1 and L2 norms are preferable choices for SIFT and SURF descriptors, NORM_HAMMING should be used with ORB, BRISK and BRIEF, NORM_HAMMING2 should be used with ORB 
"""
"""
    Devuelve el tiempo de emparejamiento y la cantidad de emparejamientos
"""
def bruteForce(img1,kp1, desc1,img2, kp2, desc2):
    start = time.time()
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_L1)
    nn_matches = matcher.knnMatch(desc1, desc2, 2)
    end = time.time()
    time_emparejamiento = end - start
#    matches = sorted(matches, key = lambda x:x.distance)
 #   img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=2)
    
    matched1 = []
    matched2 = []
    matches = []
    nn_match_ratio = 0.8 # Nearest neighbor matching ratio
    for m, n in nn_matches:
       if m.distance < nn_match_ratio * n.distance:
            matched1.append(kp1[m.queryIdx])
            matched2.append(kp2[m.trainIdx])
            matches.append([m])
    
    num_emparejamientos = len(matches)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()
    #plt.imshow(img3),plt.show()
    return time_emparejamiento, num_emparejamientos

img = cv2.imread('BuildingScene/building1.JPG')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# kp = SIFT_keypoints(gray)
# img=cv2.drawKeypoints(gray,kp,img)
# cv2.imwrite('sift_keypoints.jpg',img)


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
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    plt.imshow(img3,),plt.show()
    return time_emparejamiento, len(matches)



dst = HARRIS_keypoints(gray)
# Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.01*dst.max()]=[0,0,255]
# cv2.imshow('dst',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# kp = ORB_keypoints(gray)
# img2 = cv2.drawKeypoints(gray, kp, None, color=(0,255,0), flags=0)
# plt.imshow(img2), plt.show()

# kp = AKAZE_keypoints(gray)
# img=cv2.drawKeypoints(gray,kp,img)
# cv2.imshow('AKAZE', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


img2 = cv2.imread('BuildingScene/building2.JPG')
gray2= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kp1, desc1, time_detection, num_features = SIFT_keypoints(gray,500)
kp2, desc2, time_detection2, num_features2  = SIFT_keypoints(gray2,500)
time1, num_matches1 = flann_Matching(gray, kp1, desc1, gray2, kp2, desc2)
time2, num_matches2 = bruteForce(gray, kp1, desc1, img2, kp2, desc2)

print(time1, ' ', time2)

