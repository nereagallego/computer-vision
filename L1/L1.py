import cv2
import numpy as np
from matplotlib import pyplot as plt
#cap = cv2.VideoCapture(0)

"""_summary_ contrasta una imagen
    Args:
        img: imagen a la que se le aplica el contraste
"""
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
    return enhanced_img

"""_summary_ ecualizar una imagen
    Args:
        img: imagen que se va a ecualizar
"""
def equalizar(img):
    img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    img[:,:,2] = cv2.equalizeHist(img[:,:,2])
    return img

"""_summary_ pone una máscara de alien
    Args:
        img: imagen sobre la que se aplica la máscara
        rgbColor: imagen del mismo tamaño que la anterior con el color que se quiere aplicar a la máscara
    Returns:
        devuelve una imagen detectando la piel de una persona y cambiándolo por el color de la máscara
"""
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

    mask = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    cv2.addWeighted(img, 0.4, mask, 0.6, 1, result)

    return result

"""_summary_ reduce el numero de colores de una imagen
    Args:
        img: imagen a la que se va a hacer poster.
    Returns:
        devuelve la imagen original con el espacio de colores reducido
"""
def poster(img):
    img[img >= 170]= 255
    img[img < 85] = 0
    img[np.bitwise_and(img > 85, img < 170)] = 128
    return img

"""_summary_ Aplica distorsión radial. Depende de K1 y K2 se consiguen distorsiones de barril o cojin.
    Args:
        img: imagen que se va a distorsionar
        K1: cantidad de distorsión de pixeles al cuadrado
        K2: cantidad de distorsión de pixeles a la cuarta
    Returns:
        devuelve la imagen original con la distorsión aplicada.
"""
def barrel(img, K1: float, K2: float):
    xcent, ycent = int(img.shape[1]/2), int(img.shape[0]/2)
    map_x = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    map_y = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    for i in range(map_x.shape[0]):
        for j in range(map_x.shape[1]):
            r2 = (j - xcent) ** 2 + (i - ycent) **2
            map_x[i,j] = j + (j - xcent) * K1 * r2 + (j - xcent) * K2 * (r2 ** 2) 
            map_y[i,j] = i + (i - ycent) * K1 * r2 + (i - ycent) * K2 * (r2 ** 2)          
    dst = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    return dst


"""_summary_ Aplica filtro gaussiano a la imagen que se le pasa.
    Args:
        img: imagen a la que se va a aplicar el filtro gaussiano.
    Returns:
        devuelve la imagen original con el filtro gaussiano aplicado.
"""
def gaussianFilter(img):
    blur = cv2.GaussianBlur(img,(5,5),0)
    return blur
            

def leerOpcion():
    opcion = -1
    while opcion < 0 or opcion > 7:
        
        print('0 - salir')
        print('1 - contraste')
        print('2 - equalizacion')
        print('3 - alien')
        print('4 - poster')
        print('5 - barrel')
        print('6 - cojin')
        print('7 - gaussian filter')
        opcion = int(input())
    return opcion
cap = cv2.VideoCapture(0)
leido, img = cap.read()
img2 = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

leido = True

if leido == True:
    
    opcion = leerOpcion()
    while opcion != 0:

        if opcion == 1:
            cap = cv2.VideoCapture(0)
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                # Our operations on the frame come here
                result = contraste(frame)
                Hori = np.concatenate((frame, result), axis=1)
                # Display the resulting frame
                cv2.imshow('Contraste', Hori)
                if cv2.waitKey(1) == ord('q'):
                    break
            
        elif opcion == 2:

            cap = cv2.VideoCapture(0)
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                # Our operations on the frame come here
                result = equalizar(frame.copy())
                # Display the resulting frame
                Hori = np.concatenate((frame, result), axis=1)
                cv2.imshow('Equalization', Hori)
                if cv2.waitKey(1) == ord('q'):
                    break

        elif opcion == 3:
            
            cap = cv2.VideoCapture(0)
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                # Our operations on the frame come here
                img_copy = np.zeros(frame.shape, np.uint8);
                img_copy[:,:,2] = 255
                result = alien(frame,img_copy)
                Hori = np.concatenate((frame, result), axis=1)
                # Display the resulting frame
                cv2.imshow('Alien', Hori)
                if cv2.waitKey(1) == ord('q'):
                    break
            

        elif opcion == 4:
            
            cap = cv2.VideoCapture(0)
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                # Our operations on the frame come here
                result = poster(frame.copy())
                # Display the resulting frame
                Hori = np.concatenate((frame, result), axis=1)
                cv2.imshow('Poster frame', Hori)
                if cv2.waitKey(1) == ord('q'):
                    break
            
        elif opcion == 5:
            cap = cv2.VideoCapture(0)
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                # Our operations on the frame come here
                result = barrel(frame,-0.000005, -0.00000000005)
                # Display the resulting frame
                Hori = np.concatenate((frame, result), axis=1)
                cv2.imshow('Barrel frame', Hori)
                if cv2.waitKey(1) == ord('q'):
                    break
            
        elif opcion == 6:
            cap = cv2.VideoCapture(0)
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                # Our operations on the frame come here
                result = barrel(frame,0.0000005, -0.0000000000005)
                # Display the resulting frame
                Hori = np.concatenate((frame, result), axis=1)
                cv2.imshow("Pincushion image", Hori)
                if cv2.waitKey(1) == ord('q'):
                    break
        elif opcion == 7:
            cap = cv2.VideoCapture(0)
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                # Our operations on the frame come here
                result = gaussianFilter(frame.copy())
                # Display the resulting frame
                Hori = np.concatenate((frame, result), axis=1)
                cv2.imshow("Gaussian filter", Hori)
                if cv2.waitKey(1) == ord('q'):
                    break
            
        
        cv2.destroyAllWindows()
        
        opcion = leerOpcion()
        
else:
	print("Error al acceder a la cámara")