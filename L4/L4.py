import numpy as np
import cv2
import tarfile
from matplotlib import pyplot as plt
import random
import math

import time

# https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html

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
    

# https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
    """_summary_ Devuelve los puntos de HARRYS
        Args:

            blockSize:	Neighborhood size
            ksize:	Aperture parameter for the Sobel operator.
            k:	Harris detector free parameter.
    """
def HARRIS_keypoints(gray, blockSize=2, ksize = 3, k = 0.04):
    
    dst = cv2.cornerHarris(gray,blockSize, ksize , k)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    
    return dst

# https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html
    """_summary_ Devuelve los keypoints, los descriptores, el tiempo de detección y la cantidad de características
        Args:

            nfeatures: number of features returned
            edgeThreshold:	This is size of the border where the features are not detected. 
    """
def ORB_keypoints(gray, nfeatures, edgeThreshold = 31):

    # Initiate ORB detector
    start = time.time()
    orb = cv2.ORB_create(nfeatures,	edgeThreshold = edgeThreshold)
    # find the keypoints with ORB
    kp = orb.detect(gray,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(gray, kp)
    end = time.time()
    # draw only keypoints location,not size and orientation
    return kp, des.astype(np.float32), end - start, len(kp)


    """_summary_ Devuelve los keypoints, los descriptores, el tiempo de detección y la cantidad de características
    """
def AKAZE_keypoints(gray):
    start = time.time()
    akaze = cv2.AKAZE_create()
    kpts1, desc1 = akaze.detectAndCompute(gray, None)
    end = time.time()
    return kpts1, desc1.astype(np.float32), end - start, len(kpts1)

# https://docs.opencv.org/3.4/d3/da1/classcv_1_1BFMatcher.html
    """ _summary_ L1 and L2 norms are preferable choices for SIFT and SURF descriptors, NORM_HAMMING should be used with ORB, BRISK and BRIEF, NORM_HAMMING2 should be used with ORB 
    """
    """ _summary_
        Devuelve el tiempo de emparejamiento y la cantidad de emparejamientos
    """
def bruteForce(img1,kp1, desc1,img2, kp2, desc2):
    start = time.time()
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_L1)
    nn_matches = matcher.knnMatch(desc1, desc2, 2)
    end = time.time()
    time_emparejamiento = end - start
        
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
    return time_emparejamiento, num_emparejamientos, matches

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
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    plt.imshow(img3,),plt.show()
            
    return time_emparejamiento, len(matches), good

img = cv2.imread('BuildingScene/building1.JPG')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# kp = SIFT_keypoints(gray)
# img=cv2.drawKeypoints(gray,kp,img)
# cv2.imwrite('sift_keypoints.jpg',img)



def calculate_RANSAC_function(gray, gray2):
    kp1, desc1, time_detection, num_features = SIFT_keypoints(gray,500)
    kp2, desc2, time_detection2, num_features2  = SIFT_keypoints(gray2,500)
    time1, num_matches1, matches1 = flann_Matching(gray, kp1, desc1, gray2, kp2, desc2)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches1 ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches1 ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = gray.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img2 = cv2.polylines(gray2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    img3 = cv2.drawMatches(gray,kp1,gray2,kp2,matches1,None,**draw_params)
    plt.imshow(img3, 'gray')
    plt.show()

def calculate_RANSAC_own(gray, gray2):
    kp1, desc1, time_detection, num_features = SIFT_keypoints(gray,500)
    kp2, desc2, time_detection2, num_features2  = SIFT_keypoints(gray2,500)
    time1, num_matches1, matches1 = flann_Matching(gray, kp1, desc1, gray2, kp2, desc2)


    num_iterations = 10
    num_samples = 4
    best_model_rate = 0
    best_model = None
    best_matches_mask = None
    finished = False
    
    while not finished:

        np.random.shuffle(matches1)
        matches = matches1[:num_samples]

        src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, 0,5.0)
        matchesMask = mask.ravel().tolist()

        rest_matches = matches1[num_samples:]
        model = 0
        for m in rest_matches:
            pts = np.float32( kp1[m.queryIdx].pt ).reshape(-1,1,2)
            dst2 = cv2.perspectiveTransform(pts,M)
            pts2 = pts[0][0]
            pts2b = np.array([pts2[0], pts2[1], 1])[:,np.newaxis] 
            pts2b = np.dot(M,  pts2b)
            dst = [pts2b[0] / pts2b[2], pts2b[1] / pts2b[2]]
            dst = np.array([dst[0][0], dst[1][0]])
            # dst = dst[0][0]
            x,y = kp2[m.trainIdx].pt
            
            err = math.sqrt((dst[0] - x) ** 2 + (dst[1] - y) ** 2)
            if err < 2:
               model += 1 
        if model >= len(matches1) * 0.5:
            best_model_rate = model
            best_model = M
            best_matches_mask = matchesMask
            finished = True
            
    h,w = gray.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,best_model)
    img2 = cv2.polylines(gray2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                        # draw only inliers
                       flags = 2)
    kp12 = [kp1[m.queryIdx] for m in matches1 ]
    kp22 = [kp2[m.trainIdx] for m in matches1]
    img3 = cv2.drawMatches(gray,kp1,gray2,kp2,matches1,None,**draw_params)
    plt.imshow(img3, 'gray')
    plt.show()
            


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
gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

kp1, desc1, time_detection, num_features = SIFT_keypoints(gray,500)
kp2, desc2, time_detection2, num_features2  = SIFT_keypoints(gray2,500)
time1, num_matches1, matches1 = flann_Matching(gray, kp1, desc1, gray2, kp2, desc2)
time2, num_matches2, matches2 = bruteForce(gray, kp1, desc1, img2, kp2, desc2)

print(time1, ' ', time2)
print(num_matches1, ' ', num_matches2)

#calculate_RANSAC_function(gray, gray2)
calculate_RANSAC_own(gray, gray2)

