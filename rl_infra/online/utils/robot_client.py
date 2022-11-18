import json
from typing import Tuple

from PIL import Image
import numpy
import requests

from ...edge import config
from ...edge.utils import uncompress_nparr


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

    @staticmethod
    def getSensorReading() -> Tuple[numpy.ndarray, int]:
        imgResponse = requests.get(
            url=RobotClient.url + config.IMG_PATH,
            headers={"Content-Type": "application/octet-stream"},
        )
        if imgResponse.status_code != 200:
            raise requests.HTTPError("Failed to get image")
        img = uncompress_nparr(imgResponse.content)

        distResponse = requests.get(url=RobotClient.url + config.DIST_PATH)
        if distResponse.status_code != 200:
            raise requests.HTTPError("Failed to get distance reading")
        dist = int(distResponse.content)

        return img, dist

    # This should go with offline data collection, but I am saving here for now.
    @staticmethod
    def saveArrayAsJpeg(img: numpy.ndarray, filePath: str) -> None:
        im = Image.fromarray(img)
        im.save(filePath)

    @staticmethod
    def buzz(numberOfTones: int) -> None:
        data = {"numberOfTones": numberOfTones}
        response = requests.post(
            url=RobotClient.url + config.BUZZ_PATH,
            data=json.dumps(data),
            headers={"Content-type": "application/json"},
        )
        if response.status_code != 200:
            raise requests.HTTPError("Failed to activate buzzer")

    @staticmethod
    def blinkLed(numberOfBlinks: int) -> None:
        data = {"numberOfBlinks": numberOfBlinks}
        response = requests.post(
            url=RobotClient.url + config.LIGHT_PATH,
            data=json.dumps(data),
            headers={"Content-type": "application/json"},
        )
        if response.status_code != 200:
            raise requests.HTTPError("Failed to light LED")
