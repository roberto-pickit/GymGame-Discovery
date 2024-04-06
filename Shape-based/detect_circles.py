import cv2
import numpy as np

def is_square(approx, min_area=1000, aspect_ratio_range=(0.99, 1.01)):
    if len(approx) == 4:  
        (x, y, w, h) = cv2.boundingRect(approx)
        aspectRatio = w / float(h)
        area = w * h
        if (aspectRatio >= aspect_ratio_range[0] and aspectRatio <= aspect_ratio_range[1]) and area>min_area: 
            return True
    return False

cap = cv2.VideoCapture(0)

square_aspect_ratio_range = (0.99, 1.01)  # for perfect square
min_square_area = 1000  # minimum pixels to be considered a square

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,2,200,
                            param1=50,param2=300,minRadius=10,maxRadius=200)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
  
    edged = cv2.Canny(gray, 30, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        
        if is_square(approx, min_square_area, square_aspect_ratio_range):
            cv2.drawContours(frame, [approx], -1, (0, 0, 255), 2)

    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()