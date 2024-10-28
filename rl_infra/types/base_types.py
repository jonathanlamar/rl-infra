import base64
from dataclasses import asdict
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer
from pydantic.dataclasses import dataclass
from typing_extensions import Annotated


def compressNpArray(nparr: NDArray[Any]) -> dict[str, str | tuple[int, ...]]:
    """Returns the given numpy array as a base64 encoded string."""
    return dict(
        data=base64.b64encode(bytes(nparr)).decode("ascii"),
        shape=nparr.shape,
        dtype=str(nparr.dtype),
    )


def uncompressNpArray(data: str, shape: tuple[int, ...], dtype: str) -> NDArray[Any]:
    """Returns the given numpy array decoded from base64-encoded string."""
    dt = np.dtype(dtype)
    buff = base64.decodebytes(bytes(data, "ascii"))
    arr = np.frombuffer(buff, dtype=dt)
    return arr.reshape(shape)


@dataclass
class SerializedNumpyArray:
    data: str
    shape: tuple[int, ...]
    dtype: str


DType = TypeVar("DType")


def validateSerializedNpArray(val: Any) -> NDArray[Any]:
    res: NDArray[Any] | None = None
    arr: SerializedNumpyArray | None = None

    if isinstance(val, np.ndarray):
        res = val
    if isinstance(val, dict):
        # validate the contents of val
        arr = SerializedNumpyArray(**val)
        res = uncompressNpArray(**asdict(arr))
    if res is None:
        raise TypeError("val is not a numpy array or a serialized numpy array")
    if arr.dtype != res.dtype:  # pyright: ignore
        raise TypeError(
            f"dtype of val is incorrect.  Expected {arr.dtype}, received {res.dtype}"  # pyright: ignore
        )
    return res


NdArray = Annotated[
    np.ndarray,
    BeforeValidator(validateSerializedNpArray),
    PlainSerializer(compressNpArray, return_type=SerializedNumpyArray),
]


class NumpyArray(NDArray[Any], Generic[DType]): ...


class SerializableDataClass(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
        from_attributes=True,
        arbitrary_types_allowed=True,
    )
