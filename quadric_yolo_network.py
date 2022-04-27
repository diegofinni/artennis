import piclient
import tiny_yolo_video
import sys
import socket
from gamepacket import GamePacket
import time
import threading
from inspect import signature

first_packet = False

def sendRoutine(piClient: piclient.PiClient):
    global first_packet
    while piClient.running:
        if not first_packet:
            packet = GamePacket(-1,-1,-1,-1,-1,-1)
            piclient.sendPacket(packet)
        else:
            tiny_yolo_video.tyolo_and_nms_run()

def recvRoutine(piClient: piclient.PiClient):
    global first_packet
    while piClient.running:
        try:
            packet = piClient.recvPacket()
            if not first_packet:
                first_packet = True
            else:
                tiny_yolo_video.ballRecieved(packet)
        except Exception as e:
            print(e)
    
def main():
    if len(sys.argv) != 5:
        print("Usage: myAddr, myPort, otherAddr, otherPort")
        exit()
    
    myAddr = sys.argv[1]
    myPort = int(sys.argv[2])
    otherAddr = sys.argv[3]
    otherPort = int(sys.argv[4])

    pi = piclient.PiClient(myAddr, myPort, otherAddr, otherPort)
    pi.setSendRoutine(sendRoutine)
    pi.setRecvRoutine(recvRoutine)

    pi.run()

if __name__ == "__main__":
    main()
