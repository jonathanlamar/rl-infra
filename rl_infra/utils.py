import base64
from typing import Any

import numpy as np
from numpy.typing import NDArray


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
    buff = base64.decodebytes(bytes(data, "ascii"))  # pyright: ignore
    arr = np.frombuffer(buff, dtype=dt)
    return arr.reshape(shape)
