import cv2 
import base64
import time
import numpy as np

def getCamera():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera, exiting...")
        exit()
    return cam

cam = getCamera()
frame = None
for i in range(100):
    _, frame = cam.read()
cv2.imwrite("og.jpg", frame)

encoded,buffer = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,50])
print(buffer.shape)

msg = base64.b64encode(buffer)

data = base64.b64decode(msg,' /')
npdata = np.frombuffer(data,dtype=np.uint8)

rebuiltFrame = cv2.imdecode(npdata, 1)
cv2.imwrite("rebuilt.jpg", rebuiltFrame)

cam.release()
cv2.destroyAllWindows()