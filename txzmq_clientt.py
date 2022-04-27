#!env/bin/python3

"""
Example txzmq client.

    examples/pub_sub.py --method=bind --endpoint=ipc:///tmp/sock --mode=publisher

    examples/pub_sub.py --method=connect --endpoint=ipc:///tmp/sock --mode=subscriber
"""
import os
import sys
import time
from optparse import OptionParser
import numpy as np
import cv2
from io import BytesIO

from twisted.internet import reactor

rootdir = os.path.realpath(os.path.join(os.path.dirname(sys.argv[0]), '..'))
sys.path.append(rootdir)
os.chdir(rootdir)

from txzmq import ZmqEndpoint, ZmqFactory, ZmqPubConnection, ZmqSubConnection


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

parser = OptionParser("")
parser.add_option("-m", "--method", dest="method", help="0MQ socket connection: bind|connect")
parser.add_option("-e", "--endpoint", dest="endpoint", help="0MQ Endpoint")
parser.add_option("-M", "--mode", dest="mode", help="Mode: publisher|subscriber")
parser.set_defaults(method="connect", endpoint="epgm://wlan0;172.26.165.132:10011")

(options, args) = parser.parse_args()

zf = ZmqFactory()
e = ZmqEndpoint(options.method, "epgm://127.0.0.1;172.26.165.132:10011")
# e = ZmqEndpoint(options.method, "ipc://wlan0;localhost:5555")
# e = ZmqEndpoint(options.method, "ipc:///tmp/sock")

if options.mode == "publisher":
    s = ZmqPubConnection(zf, e)
    s.multicastRate = 10

    def publish():
        succ, im = cap.read()
        np_bytes = BytesIO()
        np.save(np_bytes, im, allow_pickle=True)
        print("publishing %r", np_bytes)
        s.publish(np_bytes.getvalue())

        reactor.callLater(.05, publish)

    publish()
else:
    s = ZmqSubConnection(zf, e)
    s.subscribe(b'')

    def doPrint(*args):
        load_bytes = BytesIO(args[0])
        loaded_np = np.load(load_bytes, allow_pickle=True)
        cv2.imshow('image', loaded_np)
        cv2.waitKey(1)

    s.gotMessage = doPrint

reactor.run()
