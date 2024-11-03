from random import randint

from rl_infra.impl.executable_utils import getParser, runTestBed
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

    runTestBed(args, random=testBed)
