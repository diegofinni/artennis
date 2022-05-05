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
        piClient.running = False
        return

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
                if (packet.ballX != 10000):
                    print("RECEIVED: ", packet)
                if (mediapipe_pose_est.ballRecieved(packet)):
                    return
        except Exception as e:
            print(e)
    
def main():
    myAddr = ""
    myPort = 0
    otherAddr = ""
    otherPort = 0
    if len(sys.argv) != 5:
        print("Default networking")
        myAddr = "172.26.165.132"
        myPort = 6666
        otherAddr = "172.26.165.64"
        otherPort = 5555
        # print("Usage: myAddr, myPort, otherAddr, otherPort")
        # exit()
    else: 
        print("Using argv for networking")
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
