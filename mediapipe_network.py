from piclient import PiClient
import mediapipe_pose_est
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
    while not first_packet:
        print("waiting")
        packet = GamePacket(10000,10000,10000,10000,10000,10000)
        piClient.sendPacket(packet)
        time.sleep(1)
    while piClient.running:
        mediapipe_pose_est.mediapipe_pose_est(piClient)

def recvRoutine(piClient: PiClient):
    global first_packet
    while piClient.running:
        try:
            packet = piClient.recvPacket()
            if not first_packet:
                # the_time = datetime.datetime.now()
                # curr_time = the_time.second*1000000 + the_time.microsecond
                first_packet = True
                # packet_second = packet.minY
                # packet_microsecond = packet.maxY
                # print(curr_time - packet_second*1000000 - packet_microsecond)
                # time.sleep(1000)
            else:
                print("RECEIVED: ", packet)
                mediapipe_pose_est.ballRecieved(packet)
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

    pi = PiClient(myAddr, myPort, otherAddr, otherPort)
    pi.setSendRoutine(sendRoutine)
    pi.setRecvRoutine(recvRoutine)

    pi.run()

if __name__ == "__main__":
    main()
