from online.impl import RobotState
import picamera
import picamera.array


class RobotDriver:
    def __init__(self) -> None:
        self.camera = picamera.PiCamera()
        self.camera.resolution = (640, 480)


    def takeSensorReading(self) -> RobotState:
        with picamera.array.PiRGBArray(self.camera) as output:
            self.camera.capture(output, 'rgb')
            rgbArray = output.array
        raise NotImplementedError
