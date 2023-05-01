import numpy as np
import cv2
import tarfile
from matplotlib import pyplot as plt
import random
import math

import time
import os

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
    return M


def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-1])
    #crop left
    elif not np.sum(frame[:,0]):
        return trim(frame[:,1:]) 
    #crop right
    elif not np.sum(frame[:,-1]):
        return trim(frame[:,:-1])    
    return frame


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
        if model >= len(matches1) * 0.2:
            best_model_rate = model
            best_model = M
            best_matches_mask = matchesMask
            finished = True
            
    # h,w = gray.shape
    # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    # dst = cv2.perspectiveTransform(pts,best_model)
    # img2 = cv2.polylines(gray2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #                    singlePointColor = None,
    #                     # draw only inliers
    #                    flags = 2)
    # kp12 = [kp1[m.queryIdx] for m in matches1 ]
    # kp22 = [kp2[m.trainIdx] for m in matches1]
    # img3 = cv2.drawMatches(gray,kp1,gray2,kp2,matches1,None,**draw_params)
    # plt.imshow(img3, 'gray')
    # plt.show()

    # dst = cv2.warpPerspective(gray,best_model,((gray.shape[1] + gray2.shape[1]), gray2.shape[0])) #wraped image
    # dst[0:gray2.shape[0], 0:gray2.shape[1]] = gray2 #stitched image
    # plt.imshow(dst)
    # plt.show()

    
    return best_model
    # out = warpImages(gray, gray2, best_model)
    # plt.imshow(out)
    # plt.show()


def stitch_images(img1, img2, H):
    # Obtener la forma de las imágenes de entrada
    h1, w1 = img1.shape
    h2, w2 = img2.shape

    # Proyectar las esquinas de la imagen 2 hacia la imagen 1
    corners = np.array([[0, 0], [0, h2 - 1], [w2 - 1, h2 - 1], [w2 - 1, 0]], dtype=np.float32)
    corners = np.array([corners])
    projected_corners = cv2.perspectiveTransform(corners, H)

    # Obtener el ancho total y la altura máxima de las imágenes proyectadas
    max_width = int(max(projected_corners[:, :, 0].max(), w1))
    max_height = int(max(projected_corners[:, :, 1].max(), h1))

    # Crear una matriz de destino para unir las imágenes
    result = np.zeros((max_height, max_width), dtype=np.uint8)

    # Proyectar la imagen 2 sobre la imagen 1
    result_warped = cv2.warpPerspective(img2, H, (max_width, max_height))

    # Copiar la imagen 1 en la matriz de destino
    result[0:h1, 0:w1] = img1

    # Unir la imagen 2 en la matriz de destino
    mask = result_warped != 0
    result[mask] = result_warped[mask]

    return result


def construct_panorama(gray, gray2, H):
    dst = cv2.warpPerspective(gray,H,(gray2.shape[1] + gray.shape[1], gray2.shape[0]))
    dst[0:gray2.shape[0], 0:gray2.shape[1]] = gray2
    return trim(dst)

            


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
_  = calculate_RANSAC_own(gray2, gray)

files = os.listdir('./BuildingScene')
base = cv2.imread('./BuildingScene/'+files[0])
base = cv2.cvtColor(base,cv2.COLOR_BGR2GRAY)

for i in range(1, len(files)):
    img = cv2.imread('./BuildingScene/' + files[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    H = calculate_RANSAC_own(img,base)
    base = construct_panorama(img,base, H)
    plt.imshow(base)
    plt.show()


"""
"""
files = os.listdir('./BuildingScene')
base = cv2.imread('./BuildingScene/'+files[0])
"""
files = []
files.append("BuildingScene/building1.JPG")
files.append("BuildingScene/building2.JPG")
files.append("BuildingScene/building3.JPG")
files.append("BuildingScene/building4.JPG")
files.append("BuildingScene/building5.JPG")


#Guardamos las imagenes de las que se va a crear el panorama
gray_images = []
for i in range(1, len(files) + 1):
    img = cv2.imread(files[i-1])
    new_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_images.append(new_gray)
    #cv2.imshow('Panorama '.format(i), gray_images[i-1])
    #cv2.waitKey(0)



# Se obtiene el panorama de las dos primeras imagenes
H = calculate_RANSAC_own(gray_images[1], gray_images[0])
panorama = stitch_images(gray_images[0],gray_images[1],H)


for i in range(2,len(gray_images)):
    print("itero")
    H = calculate_RANSAC_own(gray_images[i],panorama)
    panorama = stitch_images(panorama,gray_images[i],H)


cv2.imshow('Panorama Total', panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""