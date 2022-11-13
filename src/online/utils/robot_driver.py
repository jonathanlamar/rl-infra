from online.impl import RobotState


class RobotDriver:
    def __init__(self) -> None:
        # TODO: Initialize EasyGoPiGo instance here
        pass

    def takeSensorReading(self) -> RobotState:
        raise NotImplementedError
