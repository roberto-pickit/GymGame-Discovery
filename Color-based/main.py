import numpy as np
import cv2
import numpy as np
import sys
import tkinter as tk
from tkinter import filedialog

image_hsv = None

#pixel = (0,0,0) #RANDOM DEFAULT VALUE

def calculate_tolerance(hsv, x, y, size=5):
    """Calculate color tolerance based on neighborhood of pixel"""
    # Limits for x and y (to avoid out-of-bounds errors)
    x_min = max(0, x - size)
    y_min = max(0, y - size)
    x_max = min(hsv.shape[1], x + size)
    y_max = min(hsv.shape[0], y + size)

    # Neighborhood
    neighborhood = hsv[y_min:y_max, x_min:x_max]

    # Calculate mean and standard deviation
    mean, stddev = cv2.meanStdDev(neighborhood)

    return stddev.flatten() 

def check_boundaries(value, tolerance, ranges, upper_or_lower):
    if ranges == 0:
        # set the boundary for hue
        boundary = 180
    elif ranges == 1:
        # set the boundary for saturation and value
        boundary = 255

    if(value + tolerance > boundary):
        value = boundary
    elif (value - tolerance < 0):
        value = 0
    else:
        if upper_or_lower == 1:
            value = value + tolerance
        else:
            value = value - tolerance
    return value

def pick_color(event,x,y,flags,param):
    global upper
    global lower

    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]

        #HUE, SATURATION, AND VALUE (BRIGHTNESS) RANGES. TOLERANCE COULD BE ADJUSTED.
        # Set range = 0 for hue and range = 1 for saturation and brightness
        # set upper_or_lower = 1 for upper and upper_or_lower = 0 for lower
        tolerance = calculate_tolerance(image_hsv, x, y,10)

        hue_upper = check_boundaries(pixel[0], tolerance[0], 0, 1)
        hue_lower = check_boundaries(pixel[0], tolerance[0], 0, 0)
        saturation_upper = check_boundaries(pixel[1], tolerance[1], 1, 1)
        saturation_lower = check_boundaries(pixel[1], tolerance[1], 1, 0)
        value_upper = check_boundaries(pixel[2], tolerance[2], 1, 1)
        value_lower = check_boundaries(pixel[2], tolerance[2], 1, 0)

        upper =  np.array([hue_upper, saturation_upper, value_upper])
        lower =  np.array([hue_lower, saturation_lower, value_lower])
        #print(lower, upper)

        #A MONOCHROME MASK FOR GETTING A BETTER VISION OVER THE COLORS 
        image_mask = cv2.inRange(image_hsv,lower,upper)
        cv2.imshow("Mask",image_mask)
       

cap = cv2.VideoCapture(0)  # 0 is your default camera

ret, image_src = cap.read()
image_src=cv2.flip(image_src,1)
#CREATE THE HSV FROM THE BGR IMAGE
image_hsv = cv2.cvtColor(image_src,cv2.COLOR_BGR2HSV)

cv2.namedWindow('HSV2')
cv2.namedWindow('Mask')

#CALLBACK FUNCTION
cv2.setMouseCallback("HSV2", pick_color)
cv2.moveWindow("Mask", 1200, 300)
cv2.moveWindow("HSV2", 200, 300)
cv2.createTrackbar("Satisfied?", "HSV2", 0, 1, lambda x: None)
#cv2.createTrackbar("Capture again?", "HSV2", 0, 1, lambda x: None)


while True:
    ret, image_src = cap.read()
    image_src = cv2.flip(image_src,1)
    image_hsv = cv2.cvtColor(image_src,cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV2",image_hsv)

    if cv2.getTrackbarPos('Satisfied?', 'HSV2'):  # Check if the user is satisfied
        print("mouse reasons")
        break
    #if cv2.getTrackbarPos('Capture again?','HSV2'):
        
    if  cv2.waitKey(20) & 0xFF == 27:  # exit loop on pressing 'esc' or mouse click
        print("you pressed the button")
        break

cv2.destroyAllWindows()

kernel = np.ones((5,5), np.uint8)  # kernel for morphological operations

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of color in HSV    
    lower_val = lower
    upper_val = upper


    mask = cv2.inRange(hsv, lower_val, upper_val)

    # REDUCING NOISE
    # Apply Gaussian blur to the mask to reduce noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # MORPHOLOGICAL OPERATIONS
    # Dilate the mask to strengthen the boundaries of the colored object
    mask = cv2.dilate(mask, kernel, iterations=2)
    # Erode the mask to reduce noise
    mask = cv2.erode(mask, kernel, iterations=1)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask=mask)

    #findContours returns a list of contours and a hierarchical representation
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_m00=0
    if len(contours)>0:
        for i, contour in enumerate(contours):
            # get moments which gives features like center of mass, area, etc. of the blobs
            M = cv2.moments(contour)

            # Check if this contour has more mass/area than the previously recorded maximum
            if M["m00"] > max_m00:
                max_m00 = M["m00"]
                index = i

    # If a contour was found
        if index != -1:
            # Get the moments for the contour with maximum mass
            M = cv2.moments(contours[index])

            # using moments, calculate the center (x, y) of the blob
            # to avoid division by zero, ensure the moment is not zero
            if M["m00"] != 0: 
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
    
            # Now, cX and cY are the coordinates of the center of the blob
            print(f"Center of blob at: ({cX}, {cY})")

            
            # Draw the center on the image
            cv2.circle(mask, (cX, cY), 40, (255, 255, 255), 1)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    #cv2.imshow('res',res)
    if cv2.waitKey(20) & 0xFF == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
