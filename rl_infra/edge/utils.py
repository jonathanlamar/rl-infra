import io
import zlib

import numpy as np


def compress_nparr(nparr):
    """
    Returns the given numpy array as compressed bytestring,
    the uncompressed and the compressed byte size.
    """
    bytestream = io.BytesIO()
    np.save(bytestream, nparr)
    uncompressed = bytestream.getvalue()
    compressed = zlib.compress(uncompressed)
    return compressed, len(uncompressed), len(compressed)


def uncompress_nparr(bytestring):
    """ """
    return np.load(io.BytesIO(zlib.decompress(bytestring)))


# This would be nice to handle on the edge, but I don't know how to make flask
# subproc out and keep track of the PID
# def exploreRoutine():
#     distanceSensor.start_continuous()
#     goPiGo.forward()
#     while True:
#         dist = distanceSensor.read_range_continuous()
#         print(dist)
#         if dist < 75:
#             goPiGo.stop()
#             goPiGo.drive_cm(-20)
#             direction = 2 * round(random.random()) - 1
#             goPiGo.turn_degrees(direction * 20)
#             if distanceSensor.read_range_continuous() >= 30:
#                 goPiGo.forward()
