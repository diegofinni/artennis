from dataclasses import dataclass
import pickle

@dataclass
class GamePacket:

    __slots__ = ('ballX', 'ballY', 'minX', 'maxX', 'minY', 'maxY')

    ballX: int
    ballY: int
    minX: int
    maxX: int
    minY: int
    maxY: int

    @staticmethod
    def serialize(packet) -> bytes:
        return pickle.dumps(packet)
    
    @staticmethod
    def deserialize(byteArray: bytes):
        return pickle.loads(byteArray)
