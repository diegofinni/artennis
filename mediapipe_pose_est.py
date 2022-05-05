import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import time
import math
import numpy as np
from typing import Optional, Tuple, NamedTuple
from piclient import PiClient
from gamepacket import GamePacket


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



win = False
scores = [0, 0]
advantages = [False, False]
ballx = .1
bally = .1
# opponentImage = np.zeros((720, 1280))
# opponentImage = cv2.cvtColor(opponentImage, cv2.COLOR_GRAY2BGR)
opponentImage = np.zeros((720, 1280, 3), np.uint8)
counting_down = False
counting_start = 0
record_pose = False
ball_received = False
displayBall = True

def incrament_score(i):
  global scores, advantages, win

  if (scores[i] == 0):
    scores[i] = 15
  elif (scores[i] == 15):
    scores[i] = 30
  elif (scores[i] == 30):
    scores[i] = 40
  elif (scores[i] == 40):
    if (scores[0] == scores[1]):
      if (advantages[i] == True):
        win = True
        return True
      elif (advantages[(i+1)%2] == True):
        advantages[(i+1)%2] = False
      else:
        advantages[i] = True
    else:
      win = True
      return True
  return False


def physicsCalc(init_pose, final_pose, deltaT):
  global ballx, bally
  print(ballx, bally, init_pose.x, init_pose.y, init_pose.x*1280, init_pose.y*720)
  if ((init_pose.x*1280 + 250 < 1280*ballx or init_pose.x*1280 - 250 > 1280*ballx) or (init_pose.y*720 + 250 < 720*bally or init_pose.y*720 - 250 > 720*bally)):
    return True, -1, -1
  # physics

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

  return False, finalX, finalY

def reset():
  global ballx, bally, opponentImage, counting_down, counting_start, record_pose, ball_received, displayBall
  ballx = .1
  bally = .1
  opponentImage = np.zeros((720, 1280, 3), np.uint8)
  counting_down = False
  counting_start = 0
  record_pose = False
  ball_received = False
  displayBall = True

def ballRecieved(packet: GamePacket):
  global ballx, bally, opponentImage, counting_down, counting_start, record_pose, ball_received, win
  if (win):
      return True
  if (packet.ballX == 10000 and not counting_down and not record_pose and ball_received): # done receiving
    counting_down = True
    counting_start = time.time()
  elif packet.ballX == 20000:
    end = incrament_score(0)
    if end:
      win = True
    reset()
  elif packet.ballX != 10000:
    if packet.ballX != 30000:
      displayBall = False
      ball_received = True
      ballx = packet.ballX/opponentImage.shape[1]
      bally = packet.ballY/opponentImage.shape[0]
    xmin = packet.minX
    xmax = packet.maxX
    ymin = packet.minY
    ymax = packet.maxY
    hscale = 1280/416
    vscale = 720/416
    if (xmin != 0 and ymin != 0):
      opponentImage = np.zeros((720, 1280, 3), np.uint8)
      box_start = (int(xmin * hscale), int(ymin *vscale))
      box_end = (int(xmax *hscale), int(ymax*vscale))
      # txt_start = (int(xmin *hscale), int(ymin *vscale)-5)
      color = (255, 255, 255)
      opponentImage = cv2.rectangle(opponentImage, box_start, box_end, color, 5)
      # opponentImage = cv2.flip(opponentImage, 1)
  return False



def mediapipe_pose_est(piClient: PiClient):
  # Initialize function global variables
  global ballx, bally, opponentImage, counting_down, counting_start, record_pose, ball_received, displayBall, win
  val = 0
  # Init Pose Bool
  init_pose_bool = False    
  init_pose = 0
  final_pose = 0
  finalX = 0
  finalY = 0

  # Grab ball images to super-impose on images
  ball10, ball5, ball2_5 = Ball(10), Ball(5), Ball(2.5)

  # Pre-process input data or get a camera stream ready
  cam = cv2.VideoCapture(0)
  cam.set(3, 1280)
  cam.set(4, 720)

  cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
  cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

      
# Main Loop
  frame_start_time = time.time()
  record_start = 0
  with mp_pose.Pose(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5,
      model_complexity=0,
      smooth_landmarks=True) as pose:
    while cam.isOpened():
      success, image = cam.read() #Read Camera Frame
      # FPS
      FPS = round(2/(time.time() - frame_start_time))
      # Latency
      if win:
        cv2.destroyAllWindows()
        return
      mediapipe_total_latency = -1
      frame_start_time = time.time()
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
        displayBall = True
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
          counting_down = False

      # We want to record the pose data
      if record_pose:
        #Latency
        FPS //= 2
        mediapipe_latency_start = time.time()
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        results = pose.process(image)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        if (results == None or results.pose_landmarks == None):
          continue
        
        #Latency
        mediapipe_total_latency = time.time()-mediapipe_latency_start
        if not init_pose_bool:
          record_start = time.time()
          init_pose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
          current_pose = init_pose
          init_pose_bool = True
        # Flip the image horizontally for a selfie-view display.
        deltaT = time.time() - record_start
        current_pose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        packet = GamePacket(ballx, bally, minX=current_pose.x, maxX=current_pose.x,
                            minY=current_pose.y, maxY=current_pose.y)
        packet.ballX *= image.shape[1]
        packet.ballY *= image.shape[0]
        packet.minX *= image.shape[1]
        packet.minY *= image.shape[0]
        packet.maxX *= image.shape[1]
        packet.maxY *= image.shape[0]

        packet.ballX = 30000
        packet.ballY = 30000
        packet.minX = round(max(0, packet.minX-125)) 
        packet.minY = round(max(0, packet.minY-125))
        packet.maxX = round(min(image.shape[1], packet.maxX+125))
        packet.maxY = round(min(packet.maxY+125, image.shape[0]))
        piClient.sendPacket(packet)
        if (packet.ballX != 10000):
            print("SENDING: ", packet)
        if deltaT > 2:
          final_pose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
          record_pose = False
          init_pose_bool = False
          miss, tmp_ballx, tmp_bally = physicsCalc(init_pose,final_pose, deltaT)
          if miss:
            end = incrament_score(1)
            packet = GamePacket(20000, 20000, 20000, 20000, 20000, 20000)
            piClient.sendPacket(packet)
            if (end == True):
              win = True
              cv2.destroyAllWindows()
              return
            reset()
            continue
          else:
            ballx = tmp_ballx
            bally = tmp_bally
          packet = GamePacket(ballx, bally, minX=current_pose.x, maxX=current_pose.x,
                              minY=current_pose.y, maxY=current_pose.y)
          packet.ballX *= image.shape[1]
          packet.ballY *= image.shape[0]
          packet.minX *= image.shape[1]
          packet.minY *= image.shape[0]
          packet.maxX *= image.shape[1]
          packet.maxY *= image.shape[0]

          packet.ballX = round(packet.ballX)
          packet.ballY = round(packet.ballY)
          packet.minX = round(max(0, packet.minX-125)) 
          packet.minY = round(max(0, packet.minY-125))
          packet.maxX = round(min(image.shape[1], packet.maxX+125))
          packet.maxY = round(min(packet.maxY+125, image.shape[0]))
          piClient.sendPacket(packet)
          piClient.sendPacket(packet)
          if (packet.ballX != 10000):
            print("SENDING: ", packet)
          ball_received = False
          displayBall = False
      else:
        packet = GamePacket(10000, 10000, 10000, 10000, 10000, 10000)
        piClient.sendPacket(packet)


      # Super impose ball on frame
      ex = max(min(int(ballx*image.shape[1])-dim[0]//2, image.shape[1]-dim[0]), 0)
      ey = max(min(int(bally*image.shape[0])-dim[1]//2, image.shape[0]-dim[1]), 0)
      if displayBall:
        # Add Ball to image
        image[ey:ey+dim[1], ex:ex+dim[0], :] = image[ey:ey+dim[1], ex:ex+dim[0], :] * (1 - ball_alpha) + ball * ball_alpha
        opponent_image = opponentImage.copy()
      else:
        opponent_image = opponentImage.copy()
        opponent_image[ey:ey+dim[1], ex:ex+dim[0], :] = opponent_image[ey:ey+dim[1], ex:ex+dim[0], :] * (1 - ball_alpha) + ball * ball_alpha
      image = cv2.flip(image,1)
      opponent_image = cv2.flip(opponent_image, 1)
      image = cv2.putText(image, "P1: {} Adv: {}".format(scores[0], advantages[0]), (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 6, cv2.LINE_AA)
      opponent_image = cv2.putText(opponent_image, "P2: {} Adv: {}".format(scores[1], advantages[1]), (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 6, cv2.LINE_AA)
      image = cv2.putText(image, str(FPS), (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 6, cv2.LINE_AA)
      image = cv2.putText(image, str(mediapipe_total_latency), (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 6, cv2.LINE_AA)
      if counting_down:
        image = cv2.putText(image,str(3 - round(time.time() - counting_start)), (image.shape[1]//2-200, image.shape[0]//2+100), cv2.FONT_HERSHEY_SIMPLEX, 16, (255, 0, 0), 70, cv2.LINE_AA)
      # Display annotated image
      image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2), interpolation = cv2.INTER_AREA)
      opponent_image = cv2.resize(opponent_image, (opponent_image.shape[1]//2, 
                                                   opponent_image.shape[0]//2), 
                                  interpolation = cv2.INTER_AREA)
      image = cv2.hconcat([image, opponent_image])
      image = cv2.resize(image,(2560, 720), interpolation=cv2.INTER_AREA)
      cv2.imshow('Frame', image)
      val = cv2.waitKey(1)
      if win:
          cv2.destroyAllWindows()

if __name__ == "__main__":
    mediapipe_pose_est()
