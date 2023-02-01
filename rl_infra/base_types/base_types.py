from __future__ import annotations

from typing import Any, Generic, Type, TypeVar

import numpy as np
from pydantic import BaseModel
from pydantic.fields import ModelField
from typing_extensions import Self

from ..utils import SerializedNumpyArray, compress_nparr, uncompress_nparr

DType = TypeVar("DType")


class NumpyArray(np.ndarray, Generic[DType]):
    def __init__(self, arr: np.ndarray) -> None:
        super().__init__()

    @classmethod
    def __get_validators__(cls: Type[Self]):
        yield cls.validators

    @classmethod
    def validators(cls: Type[Self], val: Any, field: ModelField) -> np.ndarray:
        dtypeField = field.sub_fields[0]
        expectedDtype = np.dtype(dtypeField.type_.__args__[0])
        res: np.ndarray | None = None
        arr: SerializedNumpyArray | None = None
        if isinstance(val, np.ndarray):
            res = val
        if isinstance(val, dict):
            arr = SerializedNumpyArray(**val)
            res = uncompress_nparr(arr)
        if res is None:
            raise TypeError("val is not a numpy array or a serialized numpy array")
        if expectedDtype != res.dtype:
            raise TypeError(f"dtype of val is incorrect.  Expected {expectedDtype}, received {res.dtype}")
        return res


class SerializableDataClass(BaseModel):
    class Config:
        """pydantic config class"""

        allow_mutation = False
        use_enum_values = True
        json_encoders = {np.ndarray: lambda arr: compress_nparr(arr)}
