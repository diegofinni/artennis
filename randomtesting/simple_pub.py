# simple_pub.py
import zmq
import time

host = "*"
port = "5001"

# Creates a socket instance
context = zmq.Context()
socket = context.socket(zmq.PUB)

# Binds the socket to a predefined port on localhost
socket.bind("tcp://{}:{}".format(host, port))
time.sleep(1)

# Sends a string message
socket.send_string("hello")