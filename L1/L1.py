import cv2
import numpy as np
from matplotlib import pyplot as plt
#cap = cv2.VideoCapture(0)
def contraste(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image
    result = np.hstack((img, enhanced_img))
    return result



def equalizar(img):
    result = cv2.equalizeHist(img)
    return result


def alien(img,rgbColor):
   

    hsvColor = cv2.cvtColor(rgbColor, cv2.COLOR_BGR2HSV)
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_skin = np.array([0,38,0])
    upper_skin = np.array([50,120,255])

    # Create a mask. Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Bitwise-AND mask and original image
    result = cv2.bitwise_and(hsvColor,hsvColor, mask= mask)
    cv2.imshow('Mask',mask)
    cv2.waitKey(0)
    mask = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    cv2.addWeighted(img, 0.2, mask, 0.8, 0, result)


  #  img_hsv[:, :, 0] = (img_hsv[:, :, 0] - int(255 * 0.3333)) % 255
  #  img_hsv[:, :, 1] = (img_hsv[:, :, 1] - int(255 * 0.05)) % 255
  #  img_hsv[:, :, 2] = (img_hsv[:, :, 2] - int(255 * 0.05)) % 255


    return result

def poster(img):
    img[img >= 170]= 255
    img[img < 85] = 0
    img[np.bitwise_and(img > 85, img < 170)] = 128
    return img

def barrel(img, K1: float, K2: float):
    xcent, ycent = int(img.shape[1]/2), int(img.shape[0]/2)
    map_x = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    map_y = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    for i in range(map_x.shape[0]):
        for j in range(map_x.shape[1]):
            r2 = (j - xcent) ** 2 + (i - ycent) **2
            map_x[i,j] = j + (j - xcent) * K1 * r2 + (j - xcent) * K2 * (r2 ** 2) 
    for j in range(map_y.shape[1]):
        for i in range(map_y.shape[0]):
            r2 = (j - xcent) ** 2 + (i - ycent) **2
            map_y[i,j] = i + (i - ycent) * K1 * r2 + (i - ycent) * K2 * (r2 ** 2)

    dst = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    return dst
            

def leerOpcion():
    opcion = -1
    while opcion < 0 or opcion > 5:
        
        print('0 - salir')
        print('1 - contraste')
        print('2 - equalizacion')
        print('3 - alien')
        print('4 - poster')
        print('5 - barrel')
        opcion = int(input())
    return opcion

#leido, img = cap.read()
img = cv2.imread('color.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

leido = True

if leido == True:
    
    opcion = leerOpcion()
    while opcion != 0:
	#cv2.imwrite("foto.png", frame)
        if opcion == 1:
            
            cv2.imshow('Foto tomada', img)
           
            # converting to LAB color space
            
            result = contraste(img)
            cv2.imshow('Foto tomada 2', result)
        elif opcion == 2:
        
            cv2.imshow("Foto en gris",img2)
            plt.hist(img2.ravel(),256,[0,256]); plt.show()
            result2 = equalizar(img2)
            cv2.imshow('Foto equalizada', result2)
            plt.hist(result2.ravel(),256,[0,256]); plt.show()
        elif opcion == 3:
            img_copy = np.zeros(img.shape, np.uint8);
            img_copy[:,:,0] = 255
        #    cv2.imshow('Masked Image',img_copy)
            result = alien(img,img_copy)
        
            # display the mask and masked image
            cv2.imshow('Masked Image',result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif opcion == 4:
            result = poster(img)
            cv2.imshow('Poster image', result)
            cv2.waitKey(0)
        elif opcion == 5:
            result = barrel(img,0.00000000005, 0.000000000005)
            cv2.imshow("Barrel image", result)
            cv2.waitKey(0)
        
        cv2.waitKey()
        cv2.destroyAllWindows()
        
        opcion = leerOpcion()
        
        
        """
        cv2.convertScaleAbs(frame,1.5,0)
        cv2.imshow('Foto tomada 2', frame)
        cv2.waitKey()
        cv2.destroyAllWindows()
        print("Foto tomada correctamente")
    """
else:
	print("Error al acceder a la cámara")