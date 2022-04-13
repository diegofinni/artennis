import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import time

cap = cv2.VideoCapture(0)
record_pose = False
record_start = 0
val = 0
# Init Pose Bool
init_pose_bool = False    

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

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.

    # Draw the pose annotation on the image.
    if val == 32 and not record_pose:
      record_pose = True
      record_start = time.time()
    image.flags.writeable = True 
  

    # We want to record the pose data
    if record_pose:
      init_pose = 0
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
        print(init_pose)
        init_pose_bool = True 
      # Flip the image horizontally for a selfie-view display.
      if time.time() - record_start > 2:
        record_pose = False
        init_pose_bool = False

    # Display annotated image
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    val = cv2.waitKey(1)
cap.release()
