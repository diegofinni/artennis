from piclient import PiClient
import tiny_yolo_video
import sys
import socket
from gamepacket import GamePacket
import time
import threading
from inspect import signature
import datetime

first_packet = False

def sendRoutine(piClient: PiClient):
    global first_packet
    while piClient.running:
        if not first_packet:
            packet = GamePacket(10000,10000,10000,10000, 10000, 10000)#datetime.datetime.now().second, datetime.datetime.now().microsecond)
            piClient.sendPacket(packet)
            print('Waiting')
            time.sleep(1)
        else:
            tiny_yolo_video.tyolo_and_nms_run(piClient)
            piClient.running = False

def recvRoutine(piClient: PiClient):
    global first_packet
    while piClient.running:
        try:
            packet = piClient.recvPacket()
            if not first_packet:
                first_packet = True
            else:
                tiny_yolo_video.ballRecieved(packet)
            print('RECIEVING', packet)
        except Exception as e:
            print(e)
    
def main():
    myAddr=0
    myPort=0
    otherAddr=0
    otherPort=0
    if len(sys.argv) != 5:
        print("Usage: myAddr, myPort, otherAddr, otherPort")
        myAddr = '172.26.165.64'
        myPort = 5555
        otherAddr = '172.26.165.132'
        otherPort = 6666
    else:    
        myAddr = sys.argv[1]
        myPort = int(sys.argv[2])
        otherAddr = sys.argv[3]
        otherPort = int(sys.argv[4])

    pi = PiClient(myAddr, myPort, otherAddr, otherPort)
    pi.setSendRoutine(sendRoutine)
    pi.setRecvRoutine(recvRoutine)

    pi.run()

if __name__ == "__main__":
    main()
