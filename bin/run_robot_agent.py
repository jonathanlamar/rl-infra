#!/usr/bin/env python3
import dataclasses
import glob
import json
import os

from rl_infra.online.impl import RobotAgent, RobotEnvironment
from rl_infra.online.utils import RobotClient


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)

files = glob.glob("data/*.jpg")
files.append("data/history.json")
for file in files:
    if os.path.exists(file):
        os.remove(file)

env = RobotEnvironment(moveStepSizeCm=50, turnStepSizeDeg=45)
agent = RobotAgent()
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
