#!/usr/bin/env python3

# Run pip3 install . to get these imports on your path
from rl_infra.online.impl import RobotAgent, RobotEnvironment
from rl_infra.online.utils import RobotClient

env = RobotEnvironment(moveStepSizeCm=5, turnStepSizeDeg=15)
agent = RobotAgent()

for step in range(10):
    action = agent.chooseAction(env.currentState)
    print(f"Chose action {action}")
    outcome = env.step(action)
    # print(f"Received outcome {outcome}")
    RobotClient.saveArrayAsJpeg(outcome.newState.cameraImg, filePath=f"img{step}.jpg")
