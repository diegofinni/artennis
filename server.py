import numpy as np
import cv2 as cv
import sys

def main(proto, clientAddr, clientPort):
    client = makeSocket(proto, clientAddr, clientPort)
    cam = getCamera()

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error receiving frame from local camera, exiting...")
            break
        if cv.waitKey(1) == ord('q'):
            print("Exiting program...")
            break
        client.send_image("", frame)

    tearDown(client, cam)

def getCamera():
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera, exiting...")
        exit()
    return cam

def makeSocket(proto, clientAddr, clientPort):
    connectString = proto + "://" + clientAddr + ":" + clientPort
    return imagezmq.ImageSender(connect_to=connectString)

def tearDown(client, cam):
    client.close()
    cam.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    proto = sys.argv[1]
    clientAddr = sys.argv[2]
    clientPort = sys.argv[3]
    main()