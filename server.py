# run this program on each RPi to send an image stream
import socket
import time
from imutils.video import VideoStream
import imagezmq

sender = imagezmq.ImageSender(connect
