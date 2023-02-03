#!/usr/bin/env python3
from rl_infra.base_types import SerializableDataClass
from rl_infra.online.impl.robot import (
    RobotAction,
    RobotAgent,
    RobotEnvironment,
    RobotState,
)
from rl_infra.online.impl.robot.robot_environment import RobotTransition


class RobotEnvironmentImpl(RobotEnvironment):
    def __init__(self) -> None:
        super().__init__(moveStepSizeCm=50, turnStepSizeDeg=45)

    def getReward(self, oldState: RobotState, action: RobotAction, newState: RobotState) -> float:
        return 1


class RobotAgentImpl(RobotAgent):
    def choosePolicyAction(self, state: RobotState) -> RobotAction:
        return self.chooseExploreAction(state)


env = RobotEnvironmentImpl()
agent = RobotAgentImpl()
transitions = []

for step in range(100):
    action = agent.chooseAction(env.currentState)
    outcome = env.step(action)
    transitions.append(outcome)


class RobotEpoch(SerializableDataClass):
    transitions: list[RobotTransition]


history = RobotEpoch(transitions=transitions)

with open("data/history.json", "w") as f:
    f.write(history.json())
