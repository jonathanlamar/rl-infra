import requests

from edge import config
from edge.utils import uncompress_nparr
from online.impl import RobotAction, RobotState


class RobotClient:
    url = f"http://{config.SERVER_HOST}:{config.SERVER_PORT}"

    @staticmethod
    def sendAction(action: RobotAction):
        _ = requests.post(url=RobotClient.url + config.MOVE_PATH, data=str(action))

    @staticmethod
    def getSensorReading() -> RobotState:
        imgResponse = requests.get(
            url=RobotClient.url + config.IMG_PATH,
            headers={"Content-Type": "application/octet-stream"},
        )
        img = uncompress_nparr(imgResponse.content)

        distResponse = requests.get(url=RobotClient.url + config.DIST_PATH)
        dist = int(distResponse.content)

        return RobotState(img, dist)
