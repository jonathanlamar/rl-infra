from __future__ import annotations

import base64
from dataclasses import dataclass

import numpy as np


@dataclass
class SerializedNumpyArray:
    data: str
    shape: tuple[int, ...]
    dtype: str


def compress_nparr(nparr: np.ndarray) -> SerializedNumpyArray:
    """Returns the given numpy array as a base64 encoded string."""
    print("In compress_nparr")
    return SerializedNumpyArray(
        data=base64.b64encode(nparr).decode("ascii"),
        shape=nparr.shape,
        dtype=str(nparr.dtype),
    )


def uncompress_nparr(arr: SerializedNumpyArray) -> np.ndarray:
    """Returns the given numpy array decoded from base64-encoded string."""
    print("In uncompress_nparr")
    return np.frombuffer(base64.decodebytes(bytes(arr.data, "ascii")), dtype=np.dtype(arr.dtype)).reshape(arr.shape)
