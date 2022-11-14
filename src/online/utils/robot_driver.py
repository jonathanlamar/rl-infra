from online.impl import RobotState
import picamera
import picamera.array
from easygopigo3 import EasyGoPiGo3


class RobotDriver:
    def __init__(self) -> None:
        self.camera = picamera.PiCamera()
        self.camera.resolution = (640, 480)
        self.gpg = EasyGoPiGo3()
        self.ds = self.gpg.init_distance_sensor()

    def takeSensorReading(self) -> RobotState:
        with picamera.array.PiRGBArray(self.camera) as output:
            self.camera.capture(output, "rgb")
            rgbArray = output.array
        dist = self.ds.read_mm()

        rs = RobotState(rgbArray, dist)
        return rs

    def turnRight(self, degrees=15) -> None:
        self.gpg.turn_degrees(degrees)

    def turnLeft(self, degrees=15) -> None:
        self.gpg.turn_degrees(-degrees)

    def moveFoward(self, distance=10) -> None:
        self.gpg.drive_cm(distance)

    def moveBack(self, distance=10) -> None:
        self.gpg.drive_cm(-distance)
