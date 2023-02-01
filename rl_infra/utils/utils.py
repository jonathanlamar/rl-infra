from __future__ import annotations

import base64

import numpy as np


def compressNpArray(nparr: np.ndarray) -> dict:
    """Returns the given numpy array as a base64 encoded string."""
    return dict(
        data=base64.b64encode(nparr).decode("ascii"),
        shape=nparr.shape,
        dtype=str(nparr.dtype),
    )


def uncompressNpArray(arr: dict) -> np.ndarray:
    """Returns the given numpy array decoded from base64-encoded string."""
    return np.frombuffer(base64.decodebytes(bytes(arr["data"], "ascii")), dtype=np.dtype(arr["dtype"])).reshape(
        arr["shape"]
    )
