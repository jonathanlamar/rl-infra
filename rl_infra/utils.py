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


def uncompressNpArray(serializedArray: dict[str, str | tuple[int, ...]]) -> NDArray[Any]:
    """Returns the given numpy array decoded from base64-encoded string."""
    dt = np.dtype(serializedArray["dtype"])
    buff = base64.decodebytes(bytes(serializedArray["data"], "ascii"))  # pyright: ignore
    arr = np.frombuffer(buff, dtype=dt)
    return arr.reshape(arr["shape"])
