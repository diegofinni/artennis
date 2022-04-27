from dataclasses import dataclass
import pickle
import time

@dataclass
class GamePacket:

    size = 24

    __slots__ = ('ballX', 'ballY', 'minX', 'maxX', 'minY', 'maxY')

    ballX: int
    ballY: int
    minX: int
    maxX: int
    minY: int
    maxY: int

    @staticmethod
    def serialize(packet) -> bytes:
        buf = bytearray(24)
        buf[0:4]   = packet.ballX.to_bytes(4, 'little')
        buf[4:8]   = packet.ballY.to_bytes(4, 'little')
        buf[8:12]  = packet.minX.to_bytes(4, 'little')
        buf[12:16] = packet.maxX.to_bytes(4, 'little')
        buf[16:20] = packet.minY.to_bytes(4, 'little')
        buf[20:24] = packet.maxY.to_bytes(4, 'little')
        return buf
    
    @staticmethod
    def deserialize(buf: bytes):
        assert(len(buf) == GamePacket.size)
        ballX = int.from_bytes(buf[0:4], 'little')
        ballY = int.from_bytes(buf[4:8], 'little')
        minX = int.from_bytes(buf[8:12], 'little')
        maxX = int.from_bytes(buf[12:16], 'little')
        minY = int.from_bytes(buf[16:20], 'little')
        maxY = int.from_bytes(buf[20:24], 'little')
        return GamePacket(ballX, ballY, minX, maxX, minY, maxY)
