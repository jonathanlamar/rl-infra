from rl_infra.impl.executable_utils import getParser, runTestBed
from rl_infra.impl.random import RandomAgent
from rl_infra.stationary.testbed import StationaryBanditTestBed

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
