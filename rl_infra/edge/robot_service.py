import time

from easygopigo3 import EasyGoPiGo3
from flask import Flask, request
from flask.wrappers import Response
import picamera
import picamera.array

from . import config
from .utils import compress_nparr

camera = picamera.PiCamera()
camera.resolution = (640, 480)
goPiGo = EasyGoPiGo3()
distanceSensor = goPiGo.init_distance_sensor()
buzzer = goPiGo.init_buzzer(port="AD1")
led = goPiGo.init_led(port="AD2")
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
    resp = str(distanceSensor.read_mm())

    return Response(response=resp, status=200)


@app.route(config.MOVE_PATH, methods=["POST"])
def move():
    if request.json is None:
        return Response(response="Bad request format", status=400)

    action = request.json["action"]
    arg = int(request.json["arg"])

    if action == "move":
        goPiGo.drive_cm(arg)
        resp = "Moving {} cm".format(arg)
        status = 200
    elif action == "turn":
        goPiGo.turn_degrees(arg)
        resp = "Turning {} deg".format(arg)
        status = 200
    else:
        resp = "Bad action request"
        status = 400

    return Response(response=resp, status=status)


@app.route(config.BUZZ_PATH, methods=["POST"])
def buzz():
    if request.json is None:
        return Response(response="Bad request format", status=400)

    numberOfTones = int(request.json["numberOfTones"])

    for _ in range(numberOfTones):
        buzzer.sound(440)
        time.sleep(0.25)
        buzzer.sound_off()
        time.sleep(0.25)

    return Response(response="Done", status=200)


@app.route(config.LIGHT_PATH, methods=["POST"])
def light():
    if request.json is None:
        return Response(response="Bad request format", status=400)

    numberOfBlinks = int(request.json["numberOfBlinks"])

    for _ in range(numberOfBlinks):
        led.light_max()
        time.sleep(0.25)
        led.light_off()
        time.sleep(0.25)

    return Response(response="Done", status=200)


if __name__ == "__main__":
    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT)
