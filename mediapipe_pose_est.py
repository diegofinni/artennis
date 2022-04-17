import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import time
import math
import numpy as np

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

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
record_pose = False
record_start = 0
val = 0

# Init Pose Bool
init_pose_bool = False    
init_pose = 0
final_pose = 0
finalX = 0
finalY = 0

# Initialize Image
ball = cv2.imread(r'ball.png', cv2.IMREAD_UNCHANGED)
scale_percent = 20 # percent of original size
width = int(ball.shape[1] * scale_percent / 100)
height = int(ball.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
ball = cv2.resize(ball, dim, interpolation = cv2.INTER_AREA)
# Prepare pixel-wise alpha blending
ball_alpha = ball[..., 3] / 255.0
ball_alpha = np.repeat(ball_alpha[..., np.newaxis], 3, axis=2)
ball = ball[..., :3]

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0,
    smooth_landmarks=True) as pose:
  
  # Main Loop
  while cap.isOpened():
    success, image = cap.read() #Read Camera Frame
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    image_height, image_width, _ = image.shape

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.

    # Draw the pose annotation on the image.
    if val == 32 and not record_pose:
      record_pose = True
      record_start = time.time()
    image.flags.writeable = True 
  

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
        # print("InitPose")
        # print(init_pose.x)
        # print(init_pose.y)
        # Calculate ball
        finalX, finalY = physicsCalc(init_pose,final_pose, deltaT)
        # print("FinalPose")
        # print(finalX)
        # print(finalY)

    # Super impose ball on frame
    ex = max(min(int(finalX*image.shape[1])-width//2, image.shape[1]-width), 0)
    ey = max(min(int(finalY*image.shape[0])-height//2, image.shape[0]-height), 0)
    if not init_pose_bool:
      image[ey:ey+height, ex:ex+width, :] = image[ey:ey+height, ex:ex+width, :] * (1 - ball_alpha) + ball * ball_alpha
    # Display annotated image
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    val = cv2.waitKey(1)
cap.release()
