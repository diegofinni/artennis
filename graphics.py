import cv2
import numpy as np

def getCamera():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera, exiting...")
        exit()
    return cam

def main():
    alpha = 0.0
    ballImage = cv2.imread(r'ball.png')
    cv2.namedWindow("GrahpicsDesigning")
    cam = getCamera()
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error reading frame")
            cam.release()
            cv2.destroyAllWindows()
            break
        if cv2.waitKey(1) == ord('q'):
            print("Exiting...")
            cam.release()
            cv2.destroyAllWindows()
            break
        #cv2.imshow("GraphicsDesigning", ballImage)
        
        modImage = cv2.addWeighted(frame[150:250,150:250,:],alpha,ballImage[375:475,375:475,:],1-alpha,0)
        frame[150:250,150:250] = modImage

        cv2.imshow("GraphicsDesigning", frame)
        
if __name__ == "__main__":
    main()
