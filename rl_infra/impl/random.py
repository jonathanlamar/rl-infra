import argparse
from random import randint

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from rl_infra.stationary.agent import StationaryBanditAgent
from rl_infra.stationary.testbed import StationaryBanditTestBed
from rl_infra.types.agent import Policy
from rl_infra.types.transition import Action, Context


class RandomAgent(StationaryBanditAgent[Policy]):
    r"""
    A random stationary bandit agent.  This agent chooses actions completely at random.
    It is intended for use in testing and as a pathological example.
    """

    def __init__(self, numArms: int) -> None:
        self.policy = Policy()
        self.numArms = numArms

    def chooseAction(self, context: Context) -> Action:
        return randint(0, self.numArms - 1)

    def updatePolicy(self, **kwargs) -> None: ...


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


def makePlot(arr: NDArray[np.float64], yAxLabel: str) -> None:
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
    makePlot(rewards, "Average reward")
    if saveFiles:
        plt.savefig("Avg rewards.png", format="png")
    plt.show()


def plotHitRate(hitRate: NDArray[np.float64], saveFiles: bool) -> None:
    makePlot(hitRate, "% Optimal action")
    if saveFiles:
        plt.savefig("Optimal hitrate.png", format="png")
    plt.show()


if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()

    print(
        f"Initializing random stationary testbed with {args.num_bandits} bandits and "
        + f"{args.num_arms} arms."
    )

    testBed = StationaryBanditTestBed(
        agentClass=RandomAgent, numBandits=args.num_bandits, numArms=args.num_arms
    )

    print(f"Playing testbed for {args.num_rounds} rounds.")
    testBed.play(numRounds=args.num_rounds)

    print("Done.")

    avgRewards = testBed.getAverageRewardsVector()
    optimalHitRate = testBed.getPercentOptimalActionVector()

    plotRewards(avgRewards, args.save_files)
    plotHitRate(optimalHitRate, args.save_files)

    if args.save_files:
        np.save("avg_rewards.npy", avgRewards)
        np.save("hitrate.npy", optimalHitRate)
