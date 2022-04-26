######### Imports #############################################################

import sys
import socket
from gamepacket import GamePacket
import time
import threading
from inspect import signature

######### PiClient Implementation #############################################

"""

Constructor args:

The PiClient is meant to work in tandem with one other PiClient that is using a
different (addr, port) tuple. This means that the two tuples of addr and port
for each PiClient must be the inverse of the PiClient that it is working with

Runtime operation:

The PiClient simultaneously sends and receives messages to the other PiClient
by utilizing two different threads, one receiving, and one sending. The sending
thread is the same one where piClient.run() is called. When the send routine
ends, themain thread (sender thread) will terminate the recv thread and then
return from run()

send/recv routines:

The send and recv routines that the user specifies have specific required
signatures that are checked at runtime. Check the "Example Usage" section
of this file to see what these routines should look like

"""

class PiClient:

    def __init__(self, myAddr, myPort, otherAddr, otherPort):
        assert isinstance(myAddr, str)
        assert isinstance(myPort, int)
        assert isinstance(otherAddr, str)
        assert isinstance(otherPort, int)

        # Private fields        
        self.__sendRoutine = None
        self.__recvRoutine = None
        self.__maxBufSize = 1200
        self.__sendSocket = self.__makeSender()
        self.__otherPlayer = (otherAddr, otherPort)
        self.__recvSocket = self.__makeReceiver(myAddr, myPort)

        # Public fields
        self.running = False

######### Public Methods ######################################################

    def run(self):
        if self.__sendRoutine == None or self.__recvRoutine == None:
            print("Cannot start PiClient until user routines are set")
            exit()
        
        self.running = True
        self.__recvThread = threading.Thread(target=self.__recvRoutine,
                                             args=(self,))
        self.__recvThread.start()
        self.__sendRoutine(self)
        self.__close()

    def setSendRoutine(self, userSendRoutine):
        try:
            sig = signature(userSendRoutine)
            assert(len(sig.parameters) == 1)
            paramType = sig.parameters['piClient'].annotation
            assert(paramType == PiClient)
        except:
            print("Send routine doesn't have correct signature, exitting...")
            exit()
        self.__sendRoutine = userSendRoutine
    
    def setRecvRoutine(self, userRecvRoutine):
        try:
            sig = signature(userRecvRoutine)
            assert(len(sig.parameters) == 1)
            paramType = sig.parameters['piClient'].annotation
            assert(paramType == PiClient)
        except:
            print("Recv routine doesn't have correct signature, exitting...")
            exit()
        self.__recvRoutine = userRecvRoutine
    
    def sendPacket(self, packet: GamePacket):
        buf = GamePacket.serialize(packet)
        self.__sendSocket.sendto(buf, self.__otherPlayer)

    def recvPacket(self):
        buf, addr = self.__recvSocket.recvfrom(self.__maxBufSize)
        if addr[0] != self.__otherPlayer[0]:
            print("Unexpected sender, exiting...")
            exit()
        elif len(buf) % GamePacket.size != 0:
            print("Packet of unexpected size, exiting...")
            exit()
        buf = buf[len(buf) - GamePacket.size:]
        return GamePacket.deserialize(buf)

######### Private Methods #####################################################

    def __makeSender(self):
        return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def __makeReceiver(self, myAddr, myPort):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((myAddr, myPort))
        return sock

    def __close(self):
        self.running = False
        self.__sendSocket.close()
        self.__recvSocket.close()
        self.__recvThread.join()

######### Example Usage #######################################################

def sendRoutine(piClient: PiClient):
    counter = 0
    while counter < 10:
        packet = GamePacket(0, 0, 0, 0, 0, 0)
        piClient.sendPacket(packet)
        time.sleep(1)
        counter += 1

def recvRoutine(piClient: PiClient):
    while piClient.running:
        try:
            packet = piClient.recvPacket()
            print(packet, type(packet))
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
