import cv2
import numpy as np

def getCamera():
    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        print("Cannot open camera, exiting...")
        exit()
    return cam

def main():
    alpha = 0.0
    ball = cv2.imread(r'ball.png', cv2.IMREAD_UNCHANGED)
    scale_percent = 20 # percent of original size
    width = int(ball.shape[1] * scale_percent / 100)
    height = int(ball.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    ball = cv2.resize(ball, dim, interpolation = cv2.INTER_AREA)

    cv2.namedWindow("GrahpicsDesigning")
    cam = getCamera()
    cam.set(3, 1920)
    cam.set(4, 1080)

    # Prepare pixel-wise alpha blending
    ball_alpha = ball[..., 3] / 255.0
    ball_alpha = np.repeat(ball_alpha[..., np.newaxis], 3, axis=2)
    ball = ball[..., :3]

    # # read image
    # ballImage = cv2.imread(r'ball.png')
    # scale_percent = 20 # percent of original size
    # width = int(ballImage.shape[1] * scale_percent / 100)
    # height = int(ballImage.shape[0] * scale_percent / 100)
    # dim = (width, height)

    # # resize image
    # ballImage = cv2.resize(ballImage, dim, interpolation = cv2.INTER_CUBIC)
    
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

        # Pixel-wise alpha blending
        # Manual manipulating one eye
        ex, ey = 320, 110
        frame = cv2.flip(frame, 1)
        frame[ey:ey+height, ex:ex+width, :] = frame[ey:ey+height, ex:ex+width, :] * (1 - ball_alpha) + ball * ball_alpha
        
        # modImage = cv2.addWeighted(frame[150:250,150:250,:],alpha,ballImage[375:475,375:475,:],1-alpha,0)
        # frame[150:250,150:250] = modImage

        cv2.imshow("GraphicsDesigning", frame)
        
if __name__ == "__main__":
    main()
