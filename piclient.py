import sys
import cv2 as cv
import zmq
import time
import threading

class PiClient:

    def __init__(self, serverAddr, serverPort, clientAddr, clientPort):
        assert isinstance(serverAddr, str)
        assert isinstance(serverPort, int)
        assert isinstance(clientAddr, str)
        assert isinstance(clientPort, int)

        self.playing = False
        self.cam = self.getCamera()
        self.context = zmq.Context()
        self.sendSocket = self.makeSender(serverAddr, serverPort)
        self.recvSocket = self.makeReceiver(clientAddr, clientPort)
        self.sendThread = threading.Thread(target=self.sendRoutine)
        self.recvThread = threading.Thread(target=self.recvRoutine)

    @staticmethod
    def getCamera():
        cam = cv.VideoCapture(0)
        if not cam.isOpened():
            print("Cannot open camera, exiting...")
            exit()
        return cam

    def makeSender(self, serverAddr, serverPort):
        connectString = "tcp://{}:{}".format(serverAddr, str(serverPort))
        pub = self.context.socket(zmq.PUB)
        pub.bind(connectString)
        pub.setsockopt(zmq.CONFLATE, 1)
        time.sleep(1)
        return pub

    def makeReceiver(self, clientAddr, clientPort):
        connectString = "tcp://{}:{}".format(clientAddr, str(clientPort))
        sub = self.context.socket(zmq.SUB)
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.connect(connectString)
        sub.subscribe("")
        time.sleep(1)
        return sub
    
    def start(self):
        self.playing = True
        self.sendThread.start()
        self.recvThread.start()

    def sendRoutine(self):
        while self.playing:
            self.sendSocket.send_string("Hello World!")
            time.sleep(1)

    def recvRoutine(self):
        while self.playing:
            try:
                msg = self.recvSocket.recv_string(zmq.NOBLOCK)
                print("Received msg: ", msg)
            except zmq.ZMQError as e:
                pass

    def close(self):
        # Close threads
        self.playing = False
        self.sendThread.join()
        self.recvThread.join()

        # Close resources
        self.sendSocket.close()
        self.recvSocket.close()
        self.context.term()
        self.cam.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: serverAddr, serverPort, clientAddr, clientPort")
        exit()
    serverAddr = sys.argv[1]
    serverPort = int(sys.argv[2])
    clientAddr = sys.argv[3]
    clientPort = int(sys.argv[4])
    pi = PiClient(serverAddr, serverPort, clientAddr, clientPort)
    pi.start()
    time.sleep(10)
    pi.close()