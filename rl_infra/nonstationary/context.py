import numpy as np
from numpy.typing import NDArray
from pydantic import field_validator, model_validator
from typing_extensions import Self

from rl_infra.types.transition import Action, Context

NUM_ACTIONS = 20
FEATURE_VECTOR_DIMENSION = 3


class NonstationaryContext(Context):
    r"""
    Context for nonstationary contextual bandit problems.  For the sake of simplicity,
    I am fixing the feature vector dimension at 10.  In practice, this will be fixed
    unless we change it and reinitialize the model.
    """

    user: int
    availableActions: list[Action]
    featureVectors: NDArray[np.float64]

    @field_validator("availableActions")
    @classmethod
    def availableActionsShouldBeNonempty(cls, val: list[Action]) -> list[Action]:
        if not val:
            raise ValueError("Available actions should be a nonempty list.")
        return val

    @field_validator("featureVectors")
    @classmethod
    def featureVectorsShouldBeRightDimension(
        cls, val: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if len(val.shape) != 2:
            raise ValueError(f"Expected 2 dimensional array, got shape {val.shape}")
        if val.shape[0] != FEATURE_VECTOR_DIMENSION:
            raise ValueError(
                f"Expected {FEATURE_VECTOR_DIMENSION} number of rows, got {val.shape[0]}"
            )
        return val

    @model_validator(mode="after")
    def featureVecRowsEqualsAvailableActions(self) -> Self:
        if self.featureVectors.shape[1] != len(self.availableActions):
            raise ValueError(
                "Row count in feature vectors must match number of available actions"
            )
        return self

    @classmethod
    def randomNonstationaryContext(cls, user: int) -> "NonstationaryContext":
        randomNumberOfActions = np.random.randint(1, NUM_ACTIONS + 1)
        randomAvailableActions = sorted(
            np.random.choice(
                np.arange(NUM_ACTIONS), randomNumberOfActions, replace=False
            ).tolist()
        )

        randomFeatureVectors = np.random.normal(
            0, 1, size=(FEATURE_VECTOR_DIMENSION, randomNumberOfActions)
        )

        return NonstationaryContext(
            user=user,
            availableActions=randomAvailableActions,
            featureVectors=randomFeatureVectors,
        )
