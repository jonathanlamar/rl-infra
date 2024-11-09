from rl_infra.impl.action_value import ActionValueAgent
from rl_infra.impl.executable_utils import getParser, runTestBed
from rl_infra.impl.random import RandomAgent
from rl_infra.stationary.testbed import StationaryBanditTestBed

if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()

    print(
        f"Initializing action-value stationary testbed with {args.num_bandits} bandits "
        + f"and {args.num_arms} arms."
    )

    randomTestBed = StationaryBanditTestBed(
        agentClass=RandomAgent, numBandits=args.num_bandits, numArms=args.num_arms
    )

    actionValueTestBed = StationaryBanditTestBed(
        agentClass=ActionValueAgent,
        numBandits=args.num_bandits,
        numArms=args.num_arms,
        epsilon=0.1,
    )

    runTestBed(args, random=randomTestBed, action_value=actionValueTestBed)
