from easygopigo3 import EasyGoPiGo3
from flask import Flask, request
from flask.wrappers import Response
import picamera
import picamera.array

from edge import config
from edge.utils import compress_nparr

camera = picamera.PiCamera()
camera.resolution = (640, 480)
gpg = EasyGoPiGo3()
ds = gpg.init_distance_sensor()
app = Flask(__name__)


@app.route(config.IMG_PATH, methods=["GET"])
def sendCompressedImage():
    with picamera.array.PiRGBArray(camera) as output:
        camera.capture(output, "rgb")
        rgbArray = output.array
    resp, _, _ = compress_nparr(rgbArray)

    return Response(response=resp, status=200, mimetype="application/octet_stream")


@app.route(config.DIST_PATH, methods=["GET"])
def sendDistanceReading():
    resp = str(ds.read_mm())

    return Response(response=resp, status=200)


@app.route(config.MOVE_PATH, methods=["POST"])
def move():
    req = request
    action = int(req.data["action"])
    arg = int(req.data["arg"])

    if action == 1:
        gpg.drive_cm(arg)
        resp = "Moving forward"
        status = 200
    elif action == 2:
        gpg.drive_cm(-arg)
        resp = "Moving backward"
        status = 200
    elif action == 3:
        gpg.turn_degrees(arg)
        resp = "Turning left"
        status = 200
    elif action == 4:
        gpg.turn_degrees(-arg)
        resp = "Turning right"
        status = 200
    else:
        resp = "Bad action request"
        status = 400

    return Response(response=resp, status=status)


if __name__ == "__main__":
    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT)
