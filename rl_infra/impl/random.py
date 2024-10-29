import argparse
from random import randint

from rl_infra.stationary.agent import StationaryBanditAgent
from rl_infra.stationary.testbed import StationaryBanditTestBed
from rl_infra.types.agent import Policy
from rl_infra.types.transition import Action, Context


class RandomAgent(StationaryBanditAgent[Policy]):
    def __init__(self, num_arms: int) -> None:
        self.policy = Policy()
        self.num_arms = num_arms

    def chooseAction(self, context: Context) -> Action:
        return randint(0, self.num_arms - 1)

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

    testBed = StationaryBanditTestBed(
        num_bandits=args.num_bandits, num_arms=args.num_arms
    )
