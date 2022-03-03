import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0) # Load the first video device

if not cap.isOpened(): # Make sure the camera isn't in use
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()

    if not ret: # Issue with the frame, could be a driver problem
        print("Can't receive frame. Ending")
        break

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'): #Break/Quit with q
        break

cap.release()
cv.destroyAllWindows()
