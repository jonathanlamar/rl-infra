from typing import Protocol, TypeVar

from rl_infra.types.offline.data_service import DataService
from rl_infra.types.offline.model_service import ModelDbKey, ModelService

Model = TypeVar("Model", covariant=False, contravariant=False)
MService = TypeVar("MService", bound=ModelService)
DService = TypeVar("DService", bound=DataService)


class TrainingService(Protocol[Model, MService, DService]):
    modelService: MService
    dataService: DService
    policyModel: Model | None
    targetModel: Model | None

    def modelFactory(self) -> Model: ...

    def coldStart(self, modelTag: str) -> int: ...

    def retrainAndPublish(
        self,
        modelDbKey: ModelDbKey,
        epochNumber: int,
        batchSize: int,
        numBatches: int,
        validationEpisodeId: int | None = None,
    ) -> None: ...

    def validateOnEpisode(
        self, validationEpisodeId: int | None = None
    ) -> tuple[float, int]: ...
