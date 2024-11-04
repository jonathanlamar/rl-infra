from rl_infra.impl.action_value import ActionValueAgent
from rl_infra.impl.executable_utils import getStationaryParser, runTestBed
from rl_infra.impl.ucb import StationaryUcbAgent
from rl_infra.stationary.testbed import StationaryBanditTestBed

if __name__ == "__main__":
    parser = getStationaryParser()
    args = parser.parse_args()

    print(
        f"Initializing action-value stationary testbed with {args.num_bandits} bandits "
        + f"and {args.num_arms} arms."
    )

    actionValueTestBed = StationaryBanditTestBed(
        agentClass=ActionValueAgent,
        numBandits=args.num_bandits,
        numArms=args.num_arms,
        epsilon=0.1,
    )

    ucbTestBed = StationaryBanditTestBed(
        agentClass=StationaryUcbAgent,
        numBandits=args.num_bandits,
        numArms=args.num_arms,
        c=2,
    )

    runTestBed(args, action_value=actionValueTestBed, ucb=ucbTestBed)
