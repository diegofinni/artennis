# simple_sub.py
import zmq

host = "192.168.1.191"
port = "5001"

# Creates a socket instance
context = zmq.Context()
socket = context.socket(zmq.SUB)

# Connects to a bound socket
socket.connect("tcp://{}:{}".format(host, port))

# Subscribes to all topics
socket.subscribe("")

# Receives a string format message
socket.recv_string()