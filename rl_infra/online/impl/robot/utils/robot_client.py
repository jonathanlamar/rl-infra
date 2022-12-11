from dataclasses import dataclass
import json
from typing import Tuple

from PIL import Image
from numpy import ndarray, uint8
import requests

from .....edge import config
from .....edge.utils import uncompress_nparr


@dataclass
class RobotSensorReading:
    image: ndarray
    distance: int
    motionDetected: bool
    lightColor: Tuple[float, float, float, float]


class RobotClient:
    url = f"http://{config.SERVER_HOST}:{config.SERVER_PORT}"

    @staticmethod
    def sendAction(action: str, arg: int) -> None:
        data = {"action": action, "arg": arg}
        response = requests.post(
            url=RobotClient.url + config.MOVE_PATH,
            data=json.dumps(data),
            headers={"Content-type": "application/json"},
        )
        if response.status_code != 200:
            raise requests.HTTPError("Failed to get distance reading")
        else:
            print(response.content.decode("utf-8"))

    @staticmethod
    def getSensorReading() -> RobotSensorReading:
        img = RobotClient._getImage()
        dist = RobotClient._getDistance()
        motionDetected = RobotClient._getMotion()
        lightColor = RobotClient._getLightColorReading()

        return RobotSensorReading(
            image=img,
            distance=dist,
            motionDetected=motionDetected,
            lightColor=lightColor,
        )

    @staticmethod
    def _getImage() -> ndarray:
        imgResponse = requests.get(
            url=RobotClient.url + config.IMG_PATH,
            headers={"Content-Type": "application/octet-stream"},
        )
        if imgResponse.status_code != 200:
            raise requests.HTTPError("Failed to get image")

        return uncompress_nparr(imgResponse.content)

    @staticmethod
    def _getDistance() -> int:
        distResponse = requests.get(url=RobotClient.url + config.DIST_PATH)
        if distResponse.status_code != 200:
            raise requests.HTTPError("Failed to get distance reading")
        return int(distResponse.content)

    @staticmethod
    def _getMotion() -> bool:
        motionResponse = requests.get(url=RobotClient.url + config.MOTION_PATH)
        if motionResponse.status_code != 200:
            raise requests.HTTPError("Failed to get motion reading")
        return motionResponse.content == "True"

    @staticmethod
    def _getLightColorReading() -> Tuple[float, float, float, float]:
        lightColorResponse = requests.get(url=RobotClient.url + config.LIGHT_COLOR_PATH)
        if lightColorResponse.status_code != 200:
            raise requests.HTTPError("Failed to get light and color reading")
        return lightColorResponse.content

    @staticmethod
    def _rotateMast(heading: int):
        if heading < 0 or heading > 180:
            raise ValueError("Impossible heading")
        data = {"heading": heading}
        response = requests.post(
            url=RobotClient.url + config.SERVO_PATH,
            data=json.dumps(data),
            headers={"Content-type": "application/json"},
        )
        if response.status_code != 200:
            raise requests.HTTPError("Failed to get distance reading")
        else:
            print(response.content.decode("utf-8"))

    @staticmethod
    def saveArrayAsJpeg(img: ndarray, filePath: str) -> None:
        if img.dtype != uint8:
            img = img.astype(uint8)
        im = Image.fromarray(img)
        im.save(filePath)
