import numpy as np
import cv2

cap = cv2.VideoCapture(0)  # 0 is your default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of color in HSV
    lower_green = np.array([35, 100, 100], dtype=np.uint8)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)

    lower_red2 = np.array([170,120,70])
    upper_red2 = np.array([180,255,255])

    lower_skin_pink = np.array([2, 139, 124], dtype = "uint8")    
    upper_skin_pink = np.array([22, 159, 204], dtype = "uint8")

    lower_bracciale = np.array([44,25,98])
    upper_bracciale = np.array([64,45,180])
#pigiama [ 95 165  62] [115 185 142]
    
    lower_val = lower_green
    upper_val = upper_green
    # Threshold the HSV image to get only defined colors
    mask = cv2.inRange(hsv, lower_val, upper_val)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()