import json
from dataclasses import dataclass
from typing import Literal, Tuple

import requests
from numpy import ndarray, uint8
from PIL import Image

from .....base_types import NumpyArray, SerializableDataClass
from .....edge import config
from .....utils import SerializedNumpyArray, uncompress_nparr


@dataclass
class RobotSensorReading(SerializableDataClass):
    image: NumpyArray[Literal["int64"]]
    distanceSweep: NumpyArray[Literal["int64"]]
    motionDetected: bool
    lightColorSweep: NumpyArray[Literal["int64"]]


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
        sensorSweep = RobotClient._getSensorSweep()
        motionDetected = RobotClient._getMotion()

        return RobotSensorReading(
            image=img,
            distanceSweep=sensorSweep[:, 0],
            motionDetected=motionDetected,
            lightColorSweep=sensorSweep[:, 1:],
        )

    @staticmethod
    def _getImage() -> ndarray:
        imgResponse = requests.get(
            url=RobotClient.url + config.IMG_PATH,
            headers={"Content-Type": "application/json"},
        )
        if imgResponse.status_code != 200:
            raise requests.HTTPError("Failed to get image")

        return uncompress_nparr(SerializedNumpyArray(**imgResponse.json()))

    @staticmethod
    def _getDistance() -> int:
        distResponse = requests.get(url=RobotClient.url + config.DIST_PATH)
        if distResponse.status_code != 200:
            raise requests.HTTPError("Failed to get distance reading")
        return int(distResponse.content)

    @staticmethod
    def _getMotion() -> bool:
        motionResponse = requests.get(url=RobotClient.url + config.MOTION_PATH)
        content = motionResponse.content.decode("utf-8")
        if motionResponse.status_code != 200:
            raise requests.HTTPError("Failed to get motion reading")
        if content not in ["True", "False"]:
            raise RuntimeError("Invalid response")

        return content == "True"

    @staticmethod
    def _getLightColorReading() -> Tuple[float, float, float, float]:
        lightColorResponse = requests.get(
            url=RobotClient.url + config.LIGHT_COLOR_PATH,
            headers={"Content-Type": "application/json"},
        )
        if lightColorResponse.status_code != 200:
            raise requests.HTTPError("Failed to get light and color reading")

        data = uncompress_nparr(SerializedNumpyArray(**lightColorResponse.json()))
        return tuple(data)

    @staticmethod
    def _getSensorSweep() -> ndarray:
        sweepResponse = requests.get(
            url=RobotClient.url + config.SWEEP_PATH,
            headers={"Content-Type": "application/json"},
        )
        if sweepResponse.status_code != 200:
            raise requests.HTTPError("Failed to get distance sweep")

        return uncompress_nparr(SerializedNumpyArray(*sweepResponse.json()))

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
