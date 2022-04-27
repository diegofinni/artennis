import cv2
import quadric
import numpy as np
import sys
import os
import io
import json
import time
import math
from typing import Optional, Tuple, NamedTuple
import piclient
from gamepacket import GamePacket
from piclient import PiClient


labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

COLORS = np.random.uniform(0, 255, size=(len(labels), 3))

class NmsOutputShapes:
    def __init__(self, max_allowed_boxes=160, num_classes=80):

        self.vboxes_shape = (1, 1, 4, max_allowed_boxes)
        self.vboxes_dtype = np.dtype(np.int32)

        self.vbox_ids_shape = (
            1, 1, max_allowed_boxes, num_classes)
        self.vbox_ids_dtype = np.dtype(np.int16)

        self.vbox_count_shape = (1, 1, 1, num_classes)
        self.vbox_count_dtype = np.dtype(np.int32)

        self.score_shape = (
            1, 1, max_allowed_boxes, num_classes)
        self.score_dtype = np.dtype(np.int16)

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

ballx = 0
bally = 0
opponentImage = np.zeros((720, 1280, 3), np.uint8)
counting_down = False
counting_start = 0
record_pose = False
ball_received = False
displayBall = False

def ballRecieved(packet: GamePacket):
    global ballx, bally, opponentImage, counting_down, counting_start, record_pose, ball_received, displayBall
    if (packet.ballX == 10000 and not counting_down and not record_pose and ball_received): # done receiving
        counting_down = True
        displayBall = True
        counting_start = time.time()
    elif (packet.ballX != 10000):
        ball_received = True
        displayBall = False
        ballx = packet.ballX/opponentImage.shape[1]
        bally = packet.ballY/opponentImage.shape[0]
        xmin = packet.minX
        xmax = packet.maxX
        ymin = packet.minY
        ymax = packet.maxY
        hscale = 1
        vscale = 1
        opponentImage = np.zeros((720, 1280, 3), np.uint8)
        box_start = (int(xmin * hscale), int(ymin *vscale))
        box_end = (int(xmax *hscale), int(ymax*vscale))
        txt_start = (int(xmin *hscale), int(ymin *vscale)-5)
        color = (255,255,255)
        opponentImage = cv2.rectangle(opponentImage, box_start, box_end, color, 5)

class CameraStreamInput:
    """
    Initializes a camera stream and returns it as an iterable object
    """

    def __init__(self):
        self.vid = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._index = 0

    def __iter__(self):
        """
        Creates an iterator for this container.
        """
        self._index = 0
        return self

    def __next__(self) -> Optional[Tuple[np.ndarray, int]]:
        """
        @return tuple containing current image and meta data if available, otherwise None
        """
        success, frame = self.vid.read()
        if(success):
            self._index += 1
            return (frame, self._index)
        else:
            raise StopIteration

class ANamedTuple(NamedTuple):
    x: int
    y: int

def physicsCalc(init_pose, final_pose, deltaT):
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

  return finalX, finalY

def draw_boxes(image, vboxes, vbox_count, vbox_ids, hscale=1, vscale=1):
    class_id = 0
    for count in np.nditer(vbox_count):
        if(count > 0 and count < 160):
            for i in range(count):
                box_id = vbox_ids[0, 0, i, class_id]
                xmin, xmax, ymin, ymax  = (
                    vboxes[0, 0, :, box_id] / (2**12))
                if labels[class_id] ==  "tennis racket":
                    box_start = (int(xmin * hscale), int(ymin *vscale))
                    box_end = (int(xmax *hscale), int(ymax*vscale))
                    txt_start = (int(xmin *hscale), int(ymin *vscale)-5)
                    color = COLORS[class_id]
                    cv2.rectangle(image, box_start,
                              box_end, color, 5)
        class_id += 1
    
    return image

def getRacketPose(image, vboxes, vbox_count, vbox_ids, hscale=1, vscale=1):
    class_id = 0
    x = 0
    y = 0
    for count in np.nditer(vbox_count):
        if(count > 0 and count < 160):
            for i in range(count):
                box_id = vbox_ids[0, 0, i, class_id]
                xmin, xmax, ymin, ymax  = (
                    vboxes[0, 0, :, box_id] / (2**12))
                if labels[class_id] ==  "tennis racket":
                    x = (xmin+xmax)*hscale / 2
                    y = (ymin+ymax)*vscale / 2
                    x /= image.shape[1]
                    y /= image.shape[0]
        class_id += 1
    
    return ANamedTuple(x, y)

def getRacketBox(image, vboxes, vbox_count, vbox_ids, hscale=1, vscale=1):
    class_id = 0
    x = 0
    y = 0
    for count in np.nditer(vbox_count):
        if(count > 0 and count < 160):
            for i in range(count):
                box_id = vbox_ids[0, 0, i, class_id]
                xmin, xmax, ymin, ymax  = (
                    vboxes[0, 0, :, box_id] / (2**12))
                if labels[class_id] ==  "tennis racket":
                    return xmin, xmax, ymin, ymax
        class_id += 1

def tyolo_and_nms_run(piClient: PiClient):
    # # Initialize function global variables
    global ballx, bally, opponentImage, counting_down, counting_start, record_pose, ball_received, displayBall
    val = 0
    # Init Pose Bool
    init_pose_bool = False    
    init_pose = 0
    final_pose = 0
    finalX = 0
    finalY = 0
    current_pose = ANamedTuple(0, 0)

    # Grab ball images to super-impose on images
    ball10, ball5, ball2_5 = Ball(10), Ball(5), Ball(2.5)

    # Pre-process input data or get a camera stream ready
    input_camera_stream = CameraStreamInput()

    # Inputs
    input_image_shape = (1, 1, 416, 1248)
    input_dtype = np.dtype(np.int8)

    # This is the output of the NN and input of NMS
    intermediate_shape = (1, 85, 1, 2535)
    intermediate_dtype = np.dtype(np.int32)

    # outputs
    nms_outputs = NmsOutputShapes()

    weights_host_tensor = np.fromfile(
        'tiny_yolo_epu_files/tiny_yolo_v3_epu.s.tensor0.bin', dtype=np.uint8)

    tyolo_kernel = quadric.get_kernel("./tiny_yolo_epu_files/tiny_yolo_v3_epu")
    nms_kernel = quadric.get_kernel("./nms/nms_epu")

    device_manager = quadric.DeviceManager()
    device = device_manager.get_device(
        quadric.SimulatorConfiguration(num_cores=16))
    weights_input_tensor = device.allocate_and_copy_ndarray(
        weights_host_tensor)

    # device space allocation
    # Allocate input tensor for NN
    input_device_tensor = device.allocate(input_image_shape, input_dtype)

    # Allocate output tensor of NN and input for NMS
    intermediate_device_tensor = device.allocate(
        intermediate_shape, intermediate_dtype)

    # Allocate output for NMS
    vboxes_device_tensor = device.allocate(
        nms_outputs.vboxes_shape, nms_outputs.vboxes_dtype)
    vbox_ids_device_tensor = device.allocate(
        nms_outputs.vbox_ids_shape, nms_outputs.vbox_ids_dtype)
    vbox_count_device_tensor = device.allocate(
        nms_outputs.vbox_count_shape, nms_outputs.vbox_count_dtype)
    score_tensor = device.allocate(nms_outputs.score_shape, nms_outputs.score_dtype)

    #cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty(
    #    "frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    float_to_fp32 = lambda num, num_frac: int(num * (2 ** num_frac))
    
    nms_threshold = float_to_fp32(0.5, 12)
    score_threshold = float_to_fp32(0.5, 12)
    
    #FPS numbers
    start_time = time.time()
    
    for image, frame_id in input_camera_stream:
        # FPS Caluculating
        fps_value = round(1/(time.time() - start_time))
        start_time = time.time()
        
        #Latency Calulations
        quadric_latency_total_time = -1
        
        ball = ball2_5.ballImage
        ball_alpha = ball2_5.ballAlpha
        dim = ball2_5.ballDim
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

        if record_pose:
            #Latency Measure
            quadric_latency_start_time = time.time()
            
            # Prepare the frame
            resized_orig_frame = cv2.resize(image, (416, 416))
            resized_reshaped_frame = resized_orig_frame.reshape(input_image_shape)
            # Copy image to device
            device.copy_ndarray_to_device(
                resized_reshaped_frame, input_device_tensor)
            device.load_kernel(tyolo_kernel)
            # Run YoloV3
            device.run_kernel("infer", (weights_input_tensor,
                            input_device_tensor, intermediate_device_tensor))
            # Run NMS
            device.load_kernel(nms_kernel)
            device.run_kernel("filter_boxes", (intermediate_device_tensor,
                                            vboxes_device_tensor, score_tensor, vbox_count_device_tensor, vbox_ids_device_tensor, score_threshold, nms_threshold))
            vbox_count = device.copy_ndarray_from_device(
                vbox_count_device_tensor, nms_outputs.vbox_count_shape, nms_outputs.vbox_count_dtype)
            vbox_ids = device.copy_ndarray_from_device(
                vbox_ids_device_tensor, nms_outputs.vbox_ids_shape, nms_outputs.vbox_ids_dtype)
            vboxes = device.copy_ndarray_from_device(
                vboxes_device_tensor, nms_outputs.vboxes_shape, nms_outputs.vboxes_dtype)
            image = draw_boxes(image, vboxes, vbox_count, vbox_ids, hscale=image.shape[1] / 416, vscale=image.shape[0] / 416)
            
            quadric_latency_total_time = time.time() - quadric_latency_start_time;

            if not init_pose_bool:
                init_pose = getRacketPose(image, vboxes, vbox_count, vbox_ids, hscale=image.shape[1] / 416, vscale=image.shape[0] / 416)
                init_pose_bool = True 
            deltaT = time.time() - record_start
            current_pose = getRacketPose(image, vboxes, vbox_count, vbox_ids, hscale=image.shape[1] / 416, vscale=image.shape[0] / 416)
            xmin = 0
            ymin = 0
            xmax = 0
            ymax = 0
            if current_pose.x != 0 or current_pose.y != 0:
                xmin, xmax, ymin, ymax  = getRacketBox(image, vboxes, vbox_count, vbox_ids, hscale=image.shape[1] / 416, vscale=image.shape[0] / 416)
                packet = GamePacket(int(ballx*image.shape[1]), int(bally*image.shape[0]), max(int(xmin), 0), min(int(xmax),1280), max(int(ymin), 0), min(int(ymax), 720))
                print("SENDING: ", packet)
                piClient.sendPacket(packet)
            if deltaT > 2:
                final_pose = getRacketPose(image, vboxes, vbox_count, vbox_ids, hscale=image.shape[1] / 416, vscale=image.shape[0] / 416)
                if final_pose.x != 0 or current_pose.y != 0:
                    xmin, xmax, ymin, ymax  = getRacketBox(image, vboxes, vbox_count, vbox_ids, hscale=image.shape[1] / 416, vscale=image.shape[0] / 416)
                else:
                    final_pose = current_pose
                record_pose = False
                init_pose_bool = False
                ballx, bally = physicsCalc(init_pose,final_pose, deltaT)
                
                packet = GamePacket(max(min(int(ballx*image.shape[1]),0), 1280), max(min(int(bally*image.shape[0]), 0), 720), max(int(xmin), 0), min(int(xmax),1280), max(int(ymin), 0), min(int(ymax), 720))
                print("SENDING: ", packet)
                ball_received = False
                displayBall = False
        else:
            packet = GamePacket(10000, 10000, 10000, 10000, 10000, 10000)
            piClient.sendPacket(packet)

        # Super impose ball on frame
        ex = max(min(int(ballx*image.shape[1])-dim[0]//2, image.shape[1]-dim[0]), 0)
        ey = max(min(int(bally*image.shape[0])-dim[1]//2, image.shape[0]-dim[1]), 0)
        # Add Ball to image
        # imageBeforeBall = cv2.flip(cv2.resize(image, (image.shape[1]//2, image.shape[0]//2), interpolation = cv2.INTER_AREA), 1)
        if displayBall:
            image[ey:ey+dim[1], ex:ex+dim[0], :] = image[ey:ey+dim[1], ex:ex+dim[0], :] * (1 - ball_alpha) + ball * ball_alpha
            opponent_image = opponentImage.copy()
        else:
            opponent_image = opponentImage.copy()
            opponent_image[ey:ey+dim[1], ex:ex+dim[0], :] = opponent_image[ey:ey+dim[1], ex:ex+dim[0], :] * (1 - ball_alpha) + ball * ball_alpha

        image = cv2.flip(image,1)
        opponent_image = cv2.flip(opponent_image,1)
        if counting_down:
            image = cv2.putText(image,str(3 - round(time.time() - counting_start)), (image.shape[1]//2-200, image.shape[0]//2+100), cv2.FONT_HERSHEY_SIMPLEX, 16, (255, 0, 0), 70, cv2.LINE_AA)
        # Display annotated image
        image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2), interpolation = cv2.INTER_AREA)
        opponent_image = cv2.resize(opponent_image, (opponent_image.shape[1]//2, opponent_image.shape[0]//2), interpolation = cv2.INTER_AREA)
        image = cv2.hconcat([image, opponent_image])
        
        # putting the FPS count on the frame
        cv2.putText(image, str(fps_value), (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3, cv2.LINE_AA)
        # Quadric Latency Time
        cv2.putText(image, str(quadric_latency_total_time), (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow("frame", image)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
          #  exit
        val = cv2.waitKey(1)

if __name__ == "__main__":
    tyolo_and_nms_run()
