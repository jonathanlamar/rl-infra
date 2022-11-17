import json
from typing import Tuple

from PIL import Image
import numpy
import requests

from ...edge import config
from ...edge.utils import uncompress_nparr


class RobotClient:
    url = f"http://{config.CLIENT_HOST}:{config.SERVER_PORT}"

    @staticmethod
    def sendAction(action: int, arg: int) -> None:
        data = {"action": action, "arg": arg}
        print(f"Sending data {data}")
        resp = requests.post(
            url=RobotClient.url + config.MOVE_PATH,
            data=json.dumps(data),
            headers={"Content-type": "application/json"},
        )
        print(f"Response = {resp}")

    @staticmethod
    def getSensorReading() -> Tuple[numpy.ndarray, int]:
        imgResponse = requests.get(
            url=RobotClient.url + config.IMG_PATH,
            headers={"Content-Type": "application/octet-stream"},
        )
        img = uncompress_nparr(imgResponse.content)

        distResponse = requests.get(url=RobotClient.url + config.DIST_PATH)
        dist = int(distResponse.content)

        return img, dist

    # This should go with offline data collection, but I am saving here for now.
    @staticmethod
    def saveArrayAsJpeg(img: numpy.ndarray, filePath: str) -> None:
        im = Image.fromarray(img)
        im.save(filePath)
