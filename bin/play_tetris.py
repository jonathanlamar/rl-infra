#!/usr/bin/env python3
from time import sleep

from IPython.terminal.embed import embed
from pynput.keyboard import Key, KeyCode, Listener

from rl_infra.impl.tetris.offline.tetris_data_service import TetrisDataService
from rl_infra.impl.tetris.online.tetris_environment import TetrisEnvironment
from rl_infra.impl.tetris.online.tetris_transition import TetrisAction


def onPress(key: Key | KeyCode | None) -> None:
    global ACTION
    if key == Key.up:
        ACTION = TetrisAction.UP
    elif key == Key.down:
        ACTION = TetrisAction.DOWN
    elif key == Key.left:
        ACTION = TetrisAction.LEFT
    elif key == Key.right:
        ACTION = TetrisAction.RIGHT
    else:
        ACTION = TetrisAction.NONE


def mainLoop(env: TetrisEnvironment, dataService: TetrisDataService):
    global ACTION
    ACTION = TetrisAction.NONE

    while not env.gameState.dead:
        env.step(ACTION)
        ACTION = TetrisAction.NONE
        env.gameState.draw()
        sleep(0.05)

    print("Saving gameplay.")
    epochRecord = env.currentEpochRecord
    dataService.pushEpoch(epochRecord)


if __name__ == "__main__":
    dataService = TetrisDataService()

    with Listener(on_press=onPress) as listener:
        env = TetrisEnvironment(humanPlayer=True)
        mainLoop(env, dataService)
    print("You lose!")
