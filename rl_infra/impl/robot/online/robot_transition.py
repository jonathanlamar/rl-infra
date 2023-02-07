from enum import Enum

from rl_infra.impl.robot.online.robot_client import RobotSensorReading
from rl_infra.types.online.transition import Action, State, Transition


# Robot state contains no data beyond what is in the sensor reading
class RobotState(State, RobotSensorReading):
    pass


class RobotAction(Action, Enum):
    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    DO_NOTHING = "DO_NOTHING"


class RobotTransition(Transition[RobotState, RobotAction]):
    state: RobotState
    action: RobotAction
    newState: RobotState
    reward: float
    isTerminal: bool
