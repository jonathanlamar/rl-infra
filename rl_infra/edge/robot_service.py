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
distanceSensor = goPiGo.init_distance_sensor(port="AD2")
motionSensor = goPiGo.init_motion_sensor(port="AD1")
lightColorSensor = goPiGo.init_light_color_sensor()
servo = goPiGo.init_servo()


class RobotService:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.add_url_rule(
            rule=config.IMG_PATH,
            view_func=RobotService.sendCompressedImage,
            methods=["GET"],
        )
        self.app.add_url_rule(
            rule=config.DIST_PATH,
            view_func=RobotService.sendCompressedImage,
            methods=["GET"],
        )
        self.app.add_url_rule(
            rule=config.MOTION_PATH,
            view_func=RobotService.sendCompressedImage,
            methods=["GET"],
        )
        self.app.add_url_rule(
            rule=config.LIGHT_COLOR_PATH,
            view_func=RobotService.sendCompressedImage,
            methods=["GET"],
        )
        self.app.add_url_rule(
            rule=config.SERVO_PATH,
            view_func=RobotService.sendCompressedImage,
            methods=["POST"],
        )
        self.app.add_url_rule(
            rule=config.MOVE_PATH,
            view_func=RobotService.sendCompressedImage,
            methods=["POST"],
        )

    def run(self):
        self.app.run(host=config.SERVER_HOST, port=config.SERVER_PORT)

    @staticmethod
    def sendCompressedImage():
        with picamera.array.PiRGBArray(camera) as output:
            camera.capture(output, "rgb")
            rgbArray = output.array
        resp, _, _ = compress_nparr(rgbArray)

        return Response(response=resp, status=200, mimetype="application/octet_stream")

    @staticmethod
    def sendDistanceReading():
        resp = str(distanceSensor.read_mm())

        return Response(response=resp, status=200)

    @staticmethod
    def sendMotionReading():
        resp = str(motionSensor.motion_detected())

        return Response(response=resp, status=200)

    @staticmethod
    def getLightColorReading():
        resp = lightColorSensor.safe_raw_colors()

        return Response(response=resp, status=200)

    @staticmethod
    def rotateMast():
        if request.json is None:
            return Response(response="Bad request format", status=400)

        degrees = request.json["degrees"]
        if degrees < 0 or degrees > 180:
            return Response(response="Bad request value", status=400)

        servo.rotate_servo(degrees)

        return Response(
            response="Turning mast to {} degrees".format(degrees), status=200
        )

    @staticmethod
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


if __name__ == "__main__":
    service = RobotService()
    service.run()
