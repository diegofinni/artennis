import socket
import base64
import cv2 as cv
import numpy as np

MAX_BUF_SIZE = 65536

class ARSOCK:

    def __init__(localAddr, localPort, otherAddr, otherPort):
        assert isinstance(localAddr, str)
        assert isinstance(localPort, int)
        assert isinstance(otherAddr, str)
        assert isinstance(otherPort, int)

        self.sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        self.sock.bind((localAddr, localPort))
        self.otherAddr = otherAddr
        self.otherPort = otherPort

    def sendFrame(cvFrame):
        assert isinstance(cvFrame, np.ndarray)

        _, buffer = cv.imencode('.jpg', cvFrame, [cv.IMWRITE_JPEG_QUALITY,50])
        data = base64.b64encode(buffer)
        dataLen = len(data).to_bytes(4, "little")
        packet = dataLen + b':' + data
        self.sock.sendto(packet, (self.otherAddr, self.otherPort))
    
    def recvFrame() -> np.ndarray:
        packet, addr = self.sock.recvfrom(MAX_BUF_SIZE)
        if addr != self.otherAddr:
            print("Unexpected UDP traffic, exiting program...")
            exit()
        colonIdx = packet.find(b':')
        dataLen = int.from_bytes(packet[colonIdx - 4: colonIdx], "little")
        data = packet[colonIdx + 1: colonIdx + 1 + dataLen]
        data = base64.b64decode(data, ' /')
        npdata = np.frombuffer(data,dtype=np.uint8)
        return cv2.imdecode(npdata, 1)

    def close():
        self.sock.close()