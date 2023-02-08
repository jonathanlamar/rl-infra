#!/usr/bin/env python3
from __future__ import annotations

from time import sleep

from pynput.keyboard import Key, KeyCode, Listener

from rl_infra.impl.tetris.offline.services.tetris_data_service import TetrisDataService
from rl_infra.impl.tetris.online.tetris_environment import TetrisEnvironment
from rl_infra.impl.tetris.online.tetris_transition import TetrisAction
from rl_infra.types.offline.backend import SqliteConnection


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
        sleep(0.4)

    print("Saving gameplay.")
    epochRecord = env.currentEpochRecord
    dataService.pushEpoch(epochRecord)


if __name__ == "__main__":
    dataService = TetrisDataService()

    with SqliteConnection(dataService.dbPath) as cur:
        res = cur.execute("select epoch_num from data order by epoch_num desc").fetchone()
    epochIndex = res[0] + 1 if res is not None else 0

    with Listener(on_press=onPress) as listener:
        env = TetrisEnvironment(epochNumber=epochIndex)
        mainLoop(env, dataService)
    print("You lose!")
