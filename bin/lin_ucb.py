from rl_infra.impl.executable_utils import getParser, runTestBed
from rl_infra.impl.lin_ucb import LinUcbAgent
from rl_infra.nonstationary.testbed import NonstationaryBanditTestBed

if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()

    print(
        f"Initializing nonstationary lin ucb testbed with {args.num_bandits} bandits."
    )

    linUcbTestBed = NonstationaryBanditTestBed(
        agentClass=LinUcbAgent, numBandits=args.num_bandits, alpha=2
    )

    runTestBed(args, ucb=linUcbTestBed)
