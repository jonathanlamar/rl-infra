from typing import Protocol


# This one has the most uncertainty to me.  I want there to be a Policy class
# that contains the policy and methods for fitting it online and offline.  But
# I'm not sure if an agent is conceptually distinct from a policy.  Maybe we
# should iron that out in the design.
class Agent(Protocol):
    pass
