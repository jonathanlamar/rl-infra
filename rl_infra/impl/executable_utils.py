import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from rl_infra.types.testbed import TestBed


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


def _makePlot(arrays: dict[str, NDArray[np.float64]], yAxLabel: str) -> None:
    plt.figure(figsize=(10, 6))
    for lbl, arr in arrays.items():
        plt.plot(
            np.arange(1, len(arr) + 1),
            arr,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=6,
            label=lbl,
        )
    plt.xlabel("Steps")
    plt.ylabel(yAxLabel)
    plt.legend()
    plt.grid(True)


def plotRewards(rewards: dict[str, NDArray[np.float64]], saveFiles: bool) -> None:
    _makePlot(rewards, "Average reward")
    if saveFiles:
        plt.savefig("Avg rewards.png", format="png")
    plt.show()


def plotHitRate(hitRate: dict[str, NDArray[np.float64]], saveFiles: bool) -> None:
    _makePlot(hitRate, "% Optimal action")
    if saveFiles:
        plt.savefig("Optimal hitrate.png", format="png")
    plt.show()


def runTestBed(args: argparse.Namespace, **testBeds: TestBed) -> None:
    print(f"Comparing {len(testBeds)} testbeds.")
    avgRewards = {}
    optimalHitRate = {}
    for lbl, testBed in testBeds.items():
        print(f"Playing testbed {lbl} for {args.num_rounds} rounds.")
        testBed.play(numRounds=args.num_rounds)

        print("Done.")

        avgRewards[lbl] = testBed.getAverageRewardsVector()
        optimalHitRate[lbl] = testBed.getPercentOptimalActionVector()

    plotRewards(avgRewards, args.save_files)
    plotHitRate(optimalHitRate, args.save_files)

    if args.save_files:
        with open("avg_rewards.pkl", "wb") as f, open("hitrate.pkl", "wb") as g:
            pickle.dump(avgRewards, f)
            pickle.dump(optimalHitRate, g)
