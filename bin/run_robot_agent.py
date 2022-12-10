#!/usr/bin/env python3
import dataclasses
import glob
import json
import os

from rl_infra.online.impl.robot import (
    RobotAction,
    RobotAgent,
    RobotEnvironment,
    RobotState,
)
from rl_infra.online.impl.robot.utils import RobotClient


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


class RobotEnvironmentImpl(RobotEnvironment):
    def __init__(self) -> None:
        super().__init__(moveStepSizeCm=50, turnStepSizeDeg=45)

    def getReward(
        self, oldState: RobotState, action: RobotAction, newState: RobotState
    ) -> float:
        return 1


class RobotAgentImpl(RobotAgent):
    def choosePolicyAction(self, state: RobotState) -> RobotAction:
        return self.chooseExploreAction(state)


files = glob.glob("data/*.jpg")
files.append("data/history.json")
for file in files:
    if os.path.exists(file):
        os.remove(file)


env = RobotEnvironmentImpl()
agent = RobotAgentImpl()
actions = []
states = []
rewards = []

for step in range(100):
    action = agent.chooseAction(env.currentState)
    print(
        f"Current distance reading {env.currentState.distanceSensor}.\nChose action {action}"
    )
    actions.append(action)
    outcome = env.step(action)
    rewards.append(outcome.reward)
    imgPath = f"data/img{step:02}.jpg"
    RobotClient.saveArrayAsJpeg(outcome.newState.cameraImg, filePath=imgPath)
    states.append({"distance": env.currentState.distanceSensor, "imgPath": imgPath})

history = {"actions": actions, "states": states, "rewards": rewards}
with open("data/history.json", "w") as f:
    json.dump(history, f, cls=CustomJsonEncoder)
