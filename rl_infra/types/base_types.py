import base64
import json
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer
from typing_extensions import Annotated


class SerializedNumpyArray(BaseModel):
    r"""
    Pydantic dataclass representing a base64-encoded numpy array.  Also encodes its
    shape and dtype for easier deserialization.
    """

    data: str
    shape: tuple[int, ...]
    dtype: str


def compressNpArray(nparr: NDArray[Any]) -> SerializedNumpyArray:
    r"""
    Returns the given numpy array as a base64 encoded string.
    """

    return SerializedNumpyArray(
        data=base64.b64encode(bytes(nparr)).decode("ascii"),
        shape=nparr.shape,
        dtype=str(nparr.dtype),
    )


def uncompressNpArray(data: str, shape: tuple[int, ...], dtype: str) -> NDArray[Any]:
    r"""
    Returns the given numpy array decoded from base64-encoded string.
    """

    dt = np.dtype(dtype)
    buff = base64.decodebytes(bytes(data, "ascii"))
    arr = np.frombuffer(buff, dtype=dt)
    return arr.reshape(shape)


DType = TypeVar("DType")


def validateSerializedNpArray(
    val: str | dict | SerializedNumpyArray | NDArray[Any],
) -> NDArray[Any]:
    r"""
    Validates and deserializes encoded numpy arrays from various intermediate types.
    Throws if the input cannot represent a serialized numpy array.
    """

    print(f"Found {val}, which has type {type(val)}")
    if isinstance(val, np.ndarray):
        print("Found array")
        return val
    if isinstance(val, str):
        print("Found string")
        val = json.loads(val)
    if isinstance(val, dict):
        print("Found dict")
        val = SerializedNumpyArray(**val)
    if isinstance(val, SerializedNumpyArray):
        print("found SerializedNumpyArray")
        res = uncompressNpArray(**val.model_dump())
        if val.dtype != res.dtype:
            raise TypeError(
                f"dtype of res is incorrect.  Expected {val.dtype}, received {res.dtype}"
            )
        return res
    raise TypeError(f"input has invalid type {type(val)}")


NdArray = Annotated[
    np.ndarray,
    BeforeValidator(validateSerializedNpArray),
    PlainSerializer(compressNpArray, return_type=SerializedNumpyArray),
]


class NumpyArray(NDArray[Any], Generic[DType]): ...


class SerializableDataClass(BaseModel):
    r"""
    Generic serializable object.  Extends pydantic BaseModel and sets global settings.
    Allows for use of the numpy array serialization and validation.
    """

    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
        from_attributes=True,
        arbitrary_types_allowed=True,
    )
