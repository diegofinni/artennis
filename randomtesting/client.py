import sys
import cv2
import imagezmq

def main(proto, serverPort):
    server = makeSocket(proto, serverPort)
    while True:
        _, frame = server.recv_image()
        cv2.imshow("", frame)
        server.send_reply(b'OK')
        if cv2.waitKey(1) == ord('q'):
            print("Exiting program...")
            exit()

def makeSocket(proto, serverPort):
    connectString = proto + "://*:" + serverPort
    return imagezmq.ImageHub(connect_to=connectString)

if __name__ == "__main__":
    proto = sys.argv[1]
    serverPort = sys.argv[2]
    main(proto, port)