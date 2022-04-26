import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import time
import math
import numpy as np
from typing import Optional, Tuple, NamedTuple

def physicsCalc(init_pose, final_pose, deltaT):
  # physics
  print()

  # Window frame dimensions (adjust for 2m by 2m)
  width = 2 # m scale to human position (window width is 1920 pixels)
  height = 1.125 # m window height is 1080 pixels
  # turning nromalized values into window values
  init_poseModx = init_pose.x*width
  init_poseMody = (1-init_pose.y)*height
  final_poseModx = final_pose.x*width
  final_poseMody = (1-final_pose.y)*height

  # Constants
  m = 0.057 # kg mass of tennis ball
  ag = 9.8 # m/s^2 acceleration due to gravity
  Cd = 0.5 # coefficient of drag of a tennis ball
  p = 1.21 # kg/m^3 density of air
  A = 0.0034 # m^2 cross sectional area of tennis ball

  # Initial velo values
  vz0 = 13 # m/s 
  deltaZ = 10 # m (rough estimate f distance between two players)
  vx0 = (final_poseModx - init_poseModx)/deltaT # initial x velocity caused by hit impulse
  vy0 = (final_poseMody - init_poseMody)/deltaT # initial y velocity caused by hit impulse

  # Acceleration of Drag in each axis
  adx = (0.5*Cd*p*A*(vx0**2))/m
  ady = (0.5*Cd*p*A*(vy0**2))/m
  adz = (0.5*Cd*p*A*(vz0**2))/m

  # Solve for total air time based on deltaZ (Quadratic Formula)
  d = (vz0**2) - (4*(0.5*adz)*(-deltaZ)) # calculate the discriminant
  sol1 = (-vz0-math.sqrt(d))/(2*(0.5*adz)) # find two solutions
  sol2 = (-vz0+math.sqrt(d))/(2*(0.5*adz))
  totalAirTime = max(sol1, sol2) # Total air time will positive solution

  # Solve for deltaX and deltaY
  deltaX = (vx0*3)*totalAirTime + 0.5*(-adx/2)*(totalAirTime**2)
  deltaY = (vy0*3)*totalAirTime + 0.5*(-ady/2-ag/10)*(totalAirTime**2)

  # return a final normalized ball position
  finalX = init_pose.x + deltaX/width
  finalY = init_pose.y - deltaY/height

  return finalX, finalY

# class CameraStreamInput:
#     """
#     Initializes a camera stream and returns it as an iterable object
#     """

#     def __init__(self):
#         self.vid = cv2.VideoCapture(0, cv2.CAP_V4L2)
#         self.vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
#         self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#         self._index = 0

#     def __iter__(self):
#         """
#         Creates an iterator for this container.
#         """
#         self._index = 0
#         return self

#     def __next__(self) -> Optional[Tuple[np.ndarray, int]]:
#         """
#         @return tuple containing current image and meta data if available, otherwise None
#         """
#         success, frame = self.vid.read()
#         if(success):
#             self._index += 1
#             return (frame, self._index)
#         else:
#             raise StopIteration

class ANamedTuple(NamedTuple):
    x: int
    y: int

class Ball:
    ball = cv2.imread(r'ball.png', cv2.IMREAD_UNCHANGED)
    ballImage = 0
    ballAlpha = 0
    ballDim = 0
    # parameterized constructor
    def __init__(self, scale_percent):
        self.ballDim = ((int(self.ball.shape[1] * scale_percent / 100)), int(self.ball.shape[0] * scale_percent / 100))
        self.ballImage = cv2.resize(self.ball, self.ballDim, interpolation = cv2.INTER_AREA)
        self.ballAlpha = self.ballImage[..., 3] / 255.0
        self.ballAlpha = np.repeat(self.ballAlpha[..., np.newaxis], 3, axis=2)
        self.ballImage = self.ballImage[..., :3]

def mediapipe_pose_est():
  # Initialize function global variables
  record_pose = False
  record_start = 0
  val = 0
  counting_down = False
  # Init Pose Bool
  init_pose_bool = False    
  init_pose = 0
  final_pose = 0
  finalX = 0
  finalY = 0

  # Grab ball images to super-impose on images
  ball10, ball5, ball2_5 = Ball(10), Ball(5), Ball(2_5)

  # Pre-process input data or get a camera stream ready
  cam = cv2.VideoCapture(0)
  cam.set(3, 1280)
  cam.set(4, 720)

  with mp_pose.Pose(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5,
      model_complexity=0,
      smooth_landmarks=True) as pose:
    
    # Main Loop
    while cam.isOpened():
      success, image = cam.read() #Read Camera Frame
      ball = ball2_5.ballImage
      ball_alpha = ball2_5.ballAlpha
      dim = ball2_5.ballDim
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.

      # SpaceBar is hit let's annotate pose on the image.
      if val == 32 and not record_pose:
          counting_down = True # if sendOrRecieve is Recieve
          counting_start = time.time()
      image.flags.writeable = True 
      if counting_down:
          time_diff = round(time.time() - counting_start)
          if time_diff == 0:
              ball = ball10.ballImage
              ball_alpha = ball10.ballAlpha
              dim = ball10.ballDim
          elif time_diff == 1:
              ball = ball5.ballImage
              ball_alpha = ball5.ballAlpha
              dim = ball5.ballDim

          if time_diff >= 3:
              record_pose = True
              record_start = time.time()
              counting_down = False

      # We want to record the pose data
      if record_pose:
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = pose.process(image)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        if not init_pose_bool:
          init_pose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
          init_pose_bool = True 
        # Flip the image horizontally for a selfie-view display.
        deltaT = time.time() - record_start
        if deltaT > 2:
          final_pose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
          record_pose = False
          init_pose_bool = False
          finalX, finalY = physicsCalc(init_pose,final_pose, deltaT)

      # Super impose ball on frame
      ex = max(min(int(finalX*image.shape[1])-dim[0]//2, image.shape[1]-dim[0]), 0)
      ey = max(min(int(finalY*image.shape[0])-dim[1]//2, image.shape[0]-dim[1]), 0)
      # Add Ball to image
      imageBeforeBall = cv2.flip(cv2.resize(image, (image.shape[1]//2, image.shape[0]//2), interpolation = cv2.INTER_AREA), 1)
      image[ey:ey+dim[1], ex:ex+dim[0], :] = image[ey:ey+dim[1], ex:ex+dim[0], :] * (1 - ball_alpha) + ball * ball_alpha
      image = cv2.flip(image,1)
      if counting_down:
        image = cv2.putText(image,str(3 - round(time.time() - counting_start)), (image.shape[1]//2-200, image.shape[0]//2+100), cv2.FONT_HERSHEY_SIMPLEX, 16, (255, 0, 0), 70, cv2.LINE_AA)
      # Display annotated image
      image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2), interpolation = cv2.INTER_AREA)
      image = np.concatenate((image, imageBeforeBall), axis=1)
      cv2.imshow('MediaPipe Pose', image)
      val = cv2.waitKey(1)

if __name__ == "__main__":
    mediapipe_pose_est()