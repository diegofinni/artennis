######### Imports #############################################################

import sys
import zmq
import time
import threading
from inspect import signature

######### PiClient Implementation #############################################

"""

Constructor args:

The PiClient is meant to work in tandem with one other PiClient that is using a
different (addr, port) tuple. This means that the server and client tuple for
each PiClient must be the inverse of the PiClient that it is working with

Runtime operation:

The PiClient simultaneously sends and receives messages to the other PiClient by
utilizing two different threads, one receiving, and one sending. These threads
are made when start() is called, and closed when close() is called.

Socket types:

The sockets being used are PUB SUB zmq sockets. These sockets allow the PiClient
to use the zmq.CONFLATE option for its sockets. This option sets the receive
buffer to have a packet size of 1. This means that if the sender sends more than
one packet before the receiver reads anything, then all but the last packet are
dropped. This should help substantially with lag.

Network protocols:

PiClient uses the TCP protocol instead of UDP. zmq has no official support for
UDP and I have been unable to successfully use draft sockets such as the RADIO
DISH sockets that use UDP. While not allowing for packet loss will hurt
performance, I believe that having the CONFLATE option is more important
for battling lag

Routine setting:

The user must dictate what the sending and receiving routines are by writing
the functions themselves and calling the setSendRoutine() and setRecvRoutine()
functions. The signature of the routine functions must exactly match this

routine(piClient: zmq.sugar.socket.Socket) -> None:

If the signature doesn't match, the program will exit. Both routines must be
set before start is called or the program will exit. Once the routines are set
and start is called, you must simply call close before exitting. Look at the
"Example Usage" section to see what an example send and recv routine looks like

The only fields accessible by user routines are the send and recv socket and
the running boolean which is used to test if the routine should continue
running. This is set to false when close() is called

"""

class PiClient:

    def __init__(self, serverAddr, serverPort, clientAddr, clientPort):
        assert isinstance(serverAddr, str)
        assert isinstance(serverPort, int)
        assert isinstance(clientAddr, str)
        assert isinstance(clientPort, int)
        
        # Private fields
        self.__context = zmq.Context()
        self.__sendRoutine = None
        self.__recvRoutine = None

        # Public fields accessible by user routines
        self.running = False
        self.sendSocket = self.__makeSender(serverAddr, serverPort)
        self.recvSocket = self.__makeReceiver(clientAddr, clientPort)

######### Public Methods ######################################################

    def start(self):
        if self.__sendRoutine == None or self.__recvRoutine == None:
            print("Cannot start PiClient until user routines are set")
            exit()
        
        self.running = True
        self.__sendThread = threading.Thread(target=self.__sendRoutine,
                                           args=(self,))
        self.__recvThread = threading.Thread(target=self.__recvRoutine,
                                           args=(self,))
        self.__sendThread.start()
        self.__recvThread.start()

    def close(self):
        # Close threads
        self.running = False
        self.__sendThread.join()
        self.__recvThread.join()

        # Close resources
        self.sendSocket.close()
        self.recvSocket.close()
        self.__context.term()

    def setSendRoutine(self, userSendRoutine):
        try:
            sig = signature(userSendRoutine)
            assert(len(sig.parameters) == 1)
            paramType = sig.parameters['piClient'].annotation
            assert(paramType == zmq.sugar.socket.Socket)
        except:
            print("Send routine doesn't have appropriate signature, exitting...")
            exit()
        self.__sendRoutine = userSendRoutine
    
    def setRecvRoutine(self, userRecvRoutine):
        try:
            sig = signature(userRecvRoutine)
            assert(len(sig.parameters) == 1)
            paramType = sig.parameters['piClient'].annotation
            assert(paramType == zmq.sugar.socket.Socket)
        except:
            print("Recv routine doesn't have appropriate signature, exitting...")
            exit()
        self.__recvRoutine = userRecvRoutine

######### Private Methods #####################################################

    def __makeSender(self, serverAddr, serverPort):
        connectString = "tcp://{}:{}".format(serverAddr, str(serverPort))
        pub = self.__context.socket(zmq.PUB)
        pub.bind(connectString)
        pub.setsockopt(zmq.CONFLATE, 1)
        time.sleep(1)
        return pub

    def __makeReceiver(self, clientAddr, clientPort):
        connectString = "tcp://{}:{}".format(clientAddr, str(clientPort))
        sub = self.__context.socket(zmq.SUB)
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.connect(connectString)
        sub.subscribe("")
        time.sleep(1)
        return sub

######### Example Usage #######################################################

def sendRoutine(piClient: zmq.sugar.socket.Socket):
    while piClient.running:
        piClient.sendSocket.send_string("Hello World!")
        time.sleep(1)

def recvRoutine(piClient: zmq.sugar.socket.Socket):
    while piClient.running:
        try:
            msg = piClient.recvSocket.recv_string(zmq.NOBLOCK)
            print("Received msg: ", msg)
        except zmq.ZMQError as e:
            pass

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: serverAddr, serverPort, clientAddr, clientPort")
        exit()
    
    # Grab system args
    serverAddr = sys.argv[1]
    serverPort = int(sys.argv[2])
    clientAddr = sys.argv[3]
    clientPort = int(sys.argv[4])

    # Setup PiClient
    pi = PiClient(serverAddr, serverPort, clientAddr, clientPort)
    pi.setSendRoutine(sendRoutine)
    pi.setRecvRoutine(recvRoutine)

    # Run PiClient
    pi.start()
    time.sleep(10)
    pi.close()