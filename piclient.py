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

The PiClient simultaneously sends and receives messages to the other PiClient by
utilizing two different threads, one receiving, and one sending. These threads
are made when run() is called, and closed when close() is called.

Socket types:

The sockets being used are RADIO DISH zmq sockets. These sockets allow the
PiClient to use the zmq.CONFLATE option for its sockets. This option sets the
receive buffer to have a packet size of 1. This means that if the sender sends
more than one packet before the receiver reads anything, then all but the last
packet are dropped. This should help substantially with lag.

Network protocols:

PiClient uses the UDP protocol which the RADIO DISH socket types are made to
officially support. UDP was selected so that lost packets were ignored and
not retransmitted like they would be in TCP

Routine setting:

The user must dictate what the sending and receiving routines are by writing
the functions themselves and calling the setSendRoutine() and setRecvRoutine()
functions. The signature of the routine functions must exactly match this

routine(piClient: zmq.sugar.socket.Socket): -> None

If the signature doesn't match, the program will exit. Both routines must be
set before run is called or the program will exit. Once the routines are set
and run is called, you must simply call close before exitting. Look at the
"Example Usage" section to see what an example send and recv routine looks like

The only fields accessible by user routines are the send and recv socket and
the running boolean which is used to test if the routine should continue
running. This is set to false when close() is called

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

        # Public fields
        self.running = False
        self.sendSocket = self.__makeSender()
        self.otherPlayer = (otherAddr, otherPort)
        self.recvSocket = self.__makeReceiver(myAddr, myPort)

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

######### Private Methods #####################################################

    def __makeSender(self):
        return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def __makeReceiver(self, myAddr, myPort):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((myAddr, myPort))
        return sock

    def __close(self):
        self.running = False
        self.sendSocket.close()
        self.recvSocket.close()
        self.__recvThread.join()

######### Example Usage #######################################################

def sendRoutine(piClient: PiClient):
    counter = 0
    while counter < 10:
        packet = GamePacket(0, 0, 0, 0, 0, 0)
        buf = GamePacket.serialize(packet)
        piClient.sendSocket.sendto(buf, piClient.otherPlayer)
        time.sleep(1)
        counter += 1

def recvRoutine(piClient: PiClient):
    while piClient.running:
        try:
            msg, addr = piClient.recvSocket.recvfrom(1024)
            if len(msg) % 24 != 0:
                print("Corrupted maybe?")
                exit()
            else:
                msg = msg[len(msg) - 24:]
            packet = GamePacket.deserialize(msg)
            print(packet)
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
