#!/usr/bin/env python3
import dataclasses
import json

from rl_infra.online.impl import RobotAgent, RobotEnvironment
from rl_infra.online.utils import RobotClient


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


env = RobotEnvironment(moveStepSizeCm=5, turnStepSizeDeg=15)
agent = RobotAgent()
actions = []
states = []
rewards = []

for step in range(100):
    action = agent.chooseAction(env.currentState)
    actions.append(action)
    outcome = env.step(action)
    rewards.append(outcome.reward)
    imgPath = f"data/img{step}.jpg"
    RobotClient.saveArrayAsJpeg(outcome.newState.cameraImg, filePath=imgPath)
    states.append({"distance": env.currentState.distanceSensor, "imgPath": imgPath})

history = {"actions": actions, "states": states, "rewards": rewards}
with open("data/history.json", "w") as f:
    json.dump(history, f, cls=CustomJsonEncoder)