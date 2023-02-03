import json
from dataclasses import asdict
from typing import Literal

import requests
from numpy import ndarray, uint8
from PIL import Image

from rl_infra.types.base_types import NumpyArray, SerializableDataClass, SerializedNumpyArray
from rl_infra.impl.robot.edge import config
from rl_infra.utils import uncompressNpArray


class RobotSensorReading(SerializableDataClass):
    image: NumpyArray[Literal["uint8"]]
    distanceSweep: NumpyArray[Literal["int32"]]
    motionDetected: bool


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
            distanceSweep=sensorSweep,
            motionDetected=motionDetected,
        )

    @staticmethod
    def _getImage() -> ndarray:
        imgResponse = requests.get(
            url=RobotClient.url + config.IMG_PATH,
            headers={"Content-Type": "application/json"},
        )
        if imgResponse.status_code != 200:
            raise requests.HTTPError("Failed to get image")

        # Construct and deconstruct to validate contents of imgResponse
        return uncompressNpArray(asdict(SerializedNumpyArray(**imgResponse.json())))

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
    def _getLightColorReading() -> tuple[float, float, float, float]:
        lightColorResponse = requests.get(
            url=RobotClient.url + config.LIGHT_COLOR_PATH,
            headers={"Content-Type": "application/json"},
        )
        if lightColorResponse.status_code != 200:
            raise requests.HTTPError("Failed to get light and color reading")

        # Construct and deconstruct to validate contents of lightColorResponse
        data = uncompressNpArray(asdict(SerializedNumpyArray(**lightColorResponse.json())))
        return tuple(data)

    @staticmethod
    def _getSensorSweep() -> ndarray:
        sweepResponse = requests.get(
            url=RobotClient.url + config.SWEEP_PATH,
            headers={"Content-Type": "application/json"},
        )
        if sweepResponse.status_code != 200:
            raise requests.HTTPError("Failed to get distance sweep")

        # Construct and deconstruct to validate contents of lightColorResponse
        return uncompressNpArray(asdict(SerializedNumpyArray(**sweepResponse.json())))

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
