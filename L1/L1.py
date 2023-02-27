import cv2
cap = cv2.VideoCapture(0)

leido, frame = cap.read()

if leido == True:
	#cv2.imwrite("foto.png", frame)
    cv2.imshow('Foto tomada', frame)
  
    cv2.convertScaleAbs(frame,0.2,0)
    cv2.imshow('Foto tomada', frame)
    cv2.waitKey()
    cv2.destroyAllWindows()
    print("Foto tomada correctamente")
else:
	print("Error al acceder a la cámara")