import argparse

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def getParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-arms", "-a", type=int, help="Number of arms on each bandit"
    )
    parser.add_argument(
        "--num-bandits", "-b", type=int, help="Number of bandits in the testbed"
    )
    parser.add_argument("--num-rounds", "-r", type=int, help="Number of rounds to play")
    parser.add_argument("--save-files", action="store_true", default=False)

    return parser


def _makePlot(arr: NDArray[np.float64], yAxLabel: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(
        np.arange(1, len(arr) + 1),
        arr,
        marker="o",
        color="b",
        linestyle="-",
        linewidth=2,
        markersize=6,
    )
    plt.xlabel("Steps")
    plt.ylabel(yAxLabel)
    plt.grid(True)


def plotRewards(rewards: NDArray[np.float64], saveFiles: bool) -> None:
    _makePlot(rewards, "Average reward")
    if saveFiles:
        plt.savefig("Avg rewards.png", format="png")
    plt.show()


def plotHitRate(hitRate: NDArray[np.float64], saveFiles: bool) -> None:
    _makePlot(hitRate, "% Optimal action")
    if saveFiles:
        plt.savefig("Optimal hitrate.png", format="png")
    plt.show()
