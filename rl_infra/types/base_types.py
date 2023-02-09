from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Generic, Type, TypeVar

import numpy as np
from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from pydantic.fields import ModelField
from typing_extensions import Self

from rl_infra.utils import compressNpArray, uncompressNpArray

DType = TypeVar("DType")


@dataclass
class SerializedNumpyArray:
    data: str
    shape: tuple[int, ...]
    dtype: str


class NumpyArray(np.ndarray, Generic[DType]):
    @classmethod
    def __get_validators__(cls: Type[Self]):
        yield cls.validators

    @classmethod
    def validators(cls: Type[Self], val: Any, field: ModelField) -> np.ndarray:
        if field.sub_fields is None:
            raise TypeError("Sub fields not found")
        dtypeField = field.sub_fields[0]
        expectedDtype = np.dtype(dtypeField.type_.__args__[0])
        res: np.ndarray | None = None
        arr: SerializedNumpyArray | None = None
        if isinstance(val, np.ndarray):
            res = val
        if isinstance(val, dict):
            arr = SerializedNumpyArray(**val)  # validate the contents of val
            res = uncompressNpArray(asdict(arr))
        if res is None:
            raise TypeError("val is not a numpy array or a serialized numpy array")
        if expectedDtype != res.dtype:
            raise TypeError(f"dtype of val is incorrect.  Expected {expectedDtype}, received {res.dtype}")
        return res


class BasePydanticConfig:
    """pydantic config class with shared settings for all instances (unless overridden)"""

    allow_mutation = False
    use_enum_values = True
    json_encoders = {np.ndarray: lambda arr: compressNpArray(arr)}
    orm_mode = True


class SerializableDataClass(BaseModel):
    class Config(BasePydanticConfig):
        pass


class Metrics(ABC, SerializableDataClass):
    @abstractmethod
    def updateWithNewValues(self, other: Self) -> Self:
        ...

    @staticmethod
    def avgWithoutNone(num1: float | None, num2: float | None) -> float | None:
        nums = [x for x in [num1, num2] if x is not None]
        if not nums:
            return None
        return sum(nums) / len(nums)
