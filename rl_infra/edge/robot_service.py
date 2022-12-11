from easygopigo3 import EasyGoPiGo3
from flask import Flask, request
from flask.wrappers import Response
import numpy as np
import picamera
import picamera.array

from . import config
from .utils import compress_nparr


class RobotService:
    def __init__(self):
        self.app = Flask(__name__)
        self.addRules()
        self.camera = picamera.PiCamera()
        self.camera.resolution = (640, 480)
        self.goPiGo = EasyGoPiGo3()
        self.distanceSensor = self.goPiGo.init_distance_sensor(port="AD2")
        self.motionSensor = self.goPiGo.init_motion_sensor(port="AD1")
        self.lightColorSensor = self.goPiGo.init_light_color_sensor()
        self.servo = self.goPiGo.init_servo()

    def addRules(self):
        self.app.add_url_rule(
            rule=config.IMG_PATH,
            view_func=self.sendCompressedImage,
            methods=["GET"],
        )
        self.app.add_url_rule(
            rule=config.DIST_PATH,
            view_func=self.sendDistanceReading,
            methods=["GET"],
        )
        self.app.add_url_rule(
            rule=config.MOTION_PATH,
            view_func=self.sendMotionReading,
            methods=["GET"],
        )
        self.app.add_url_rule(
            rule=config.LIGHT_COLOR_PATH,
            view_func=self.getLightColorReading,
            methods=["GET"],
        )
        self.app.add_url_rule(
            rule=config.SERVO_PATH,
            view_func=self.rotateMast,
            methods=["POST"],
        )
        self.app.add_url_rule(
            rule=config.MOVE_PATH,
            view_func=self.move,
            methods=["POST"],
        )

    def run(self):
        self.app.run(host=config.SERVER_HOST, port=config.SERVER_PORT)

    def sendCompressedImage(self):
        with picamera.array.PiRGBArray(self.camera) as output:
            self.camera.capture(output, "rgb")
            rgbArray = output.array
        resp, _, _ = compress_nparr(rgbArray)

        return Response(response=resp, status=200, mimetype="application/octet_stream")

    def sendDistanceReading(self):
        resp = str(self.distanceSensor.read_mm())

        return Response(response=resp, status=200)

    def sendMotionReading(self):
        resp = str(self.motionSensor.motion_detected())

        return Response(response=resp, status=200)

    def getLightColorReading(self):
        r, g, b, a = self.lightColorSensor.safe_raw_colors()
        resp = {"R": r, "G": g, "B": b, "alpha": a}

        return Response(response=resp, status=200)

    def rotateMast(self):
        if request.json is None:
            return Response(response="Bad request format", status=400)

        heading = request.json["heading"]
        if heading < 0 or heading > 180:
            return Response(response="Bad request value", status=400)

        self.servo.rotate_servo(heading)

        return Response(
            response="Turning mast to {} degrees".format(heading), status=200
        )

    def move(self):
        if request.json is None:
            return Response(response="Bad request format", status=400)

        action = request.json["action"]
        arg = int(request.json["arg"])

        if action == "move":
            self.goPiGo.drive_cm(arg)
            resp = "Moving {} cm".format(arg)
            status = 200
        elif action == "turn":
            self.goPiGo.turn_degrees(arg)
            resp = "Turning {} deg".format(arg)
            status = 200
        else:
            resp = "Bad action request"
            status = 400

        return Response(response=resp, status=status)

    def mastSweepAndRecordDist(self):
        self.distanceSensor.start_continuous()
        distances = np.arange(181)
        for deg in range(181):
            self.servo.rotate_servo(deg)
            distances[deg] = self.distanceSensor.read_range_continuous()

        resp, _, _ = compress_nparr(distances)

        Response(response=resp, status=200, mimetype="application/octet_stream")


if __name__ == "__main__":
    service = RobotService()
    service.run()
