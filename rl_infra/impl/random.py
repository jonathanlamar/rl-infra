import argparse
from random import randint

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

    return parser


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
