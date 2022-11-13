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
            self.camera.capture(output, 'rgb')
            rgbArray = output.array
        dist = self.ds.read_mm()

        rs = RobotState(rgbArray, dist)
        return rs


    def turnRight(self, deg=15) -> None:
        self.gpg.turn_degrees(deg)

    def turnLeft(self, deg=15) -> None:
        self.gpg.turn_degrees(-deg)

    def moveFoward(self, dist=10) -> None:
        self.gpg.drive_cm(dist)

    def moveBack(self, dist=10) -> None:
        self.gpg.drive_cm(-dist)