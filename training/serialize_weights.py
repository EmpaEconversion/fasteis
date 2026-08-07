"""The `.feisnn` weight container read by src/nn.rs.

Flat self-describing little-endian binary layout:

magic     8 bytes  "FEISNN02"
n_meta    u32      then n_meta pairs of (u32 len, utf-8 bytes) key, value
n_tensors u32      then per tensor:
            u32 len + utf-8 name
            u8  dtype  (0 f32, 1 f16, 2 int8)
            u8  ndim, ndim * u32 dims
            f32 scale, int8 only
            prod(dims) elements of the tagged type

int8 is symmetric per tensor: `value = q * scale`, `scale = max|w| / 127`.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

MAGIC = b"FEISNN02"

F32, F16, INT8 = 0, 1, 2

# Weights are named `w.<layer>.<weight|bias>`, read by src/nn.rs
# Only weights get reduced from f32 -> f16 or int8, scaling etc. says f32
WEIGHT_PREFIX = "w."

DTYPES = {"f32": F32, "f16": F16, "int8": INT8}

_NUMPY = {F32: "<f4", F16: "<f2", INT8: "<i1"}


def _write_string(out: list[bytes], text: str) -> None:
    raw = text.encode("utf-8")
    out.append(struct.pack("<I", len(raw)))
    out.append(raw)


def _quantise(array: NDArray[np.floating]) -> tuple[NDArray[np.int8], float]:
    """Symmetric per-tensor int8. Zero tensors keep scale 1 rather than dividing by 0."""
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    scale = peak / 127.0 if peak > 0.0 else 1.0
    return np.clip(np.round(array / scale), -127, 127).astype(np.int8), scale


def write(
    path: str | Path,
    metadata: dict[str, str],
    tensors: dict[str, NDArray[np.floating]],
    dtype: str = "f16",
) -> None:
    """Write a weight container."""
    if dtype not in DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}, expected one of {sorted(DTYPES)}")

    out: list[bytes] = [MAGIC, struct.pack("<I", len(metadata))]
    for key, value in metadata.items():
        _write_string(out, key)
        _write_string(out, value)

    out.append(struct.pack("<I", len(tensors)))
    for name, array in tensors.items():
        tag = DTYPES[dtype] if name.startswith(WEIGHT_PREFIX) else F32  # only quantise the weights
        array = np.ascontiguousarray(array, dtype=np.float64)
        _write_string(out, name)
        out.append(struct.pack("<BB", tag, array.ndim))
        out.extend(struct.pack("<I", d) for d in array.shape)

        if tag == INT8:
            quantised, scale = _quantise(array)
            out.append(struct.pack("<f", scale))
            out.append(quantised.tobytes())
        else:
            out.append(array.astype(_NUMPY[tag]).tobytes())

    Path(path).write_bytes(b"".join(out))


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def take(self, n: int) -> bytes:
        chunk = self.data[self.pos : self.pos + n]
        if len(chunk) != n:
            raise ValueError("truncated file")
        self.pos += n
        return chunk

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u8(self) -> int:
        return self.take(1)[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.take(4))[0]

    def string(self) -> str:
        return self.take(self.u32()).decode("utf-8")


def read(path: str | Path) -> tuple[dict[str, str], dict[str, NDArray[np.float64]]]:
    """Read a weight container. Used to round-trip test the writer."""
    r = _Reader(Path(path).read_bytes())
    if r.take(len(MAGIC)) != MAGIC:
        raise ValueError("not a feisnn file")

    metadata = {}
    for _ in range(r.u32()):
        key = r.string()
        metadata[key] = r.string()

    tensors = {}
    for _ in range(r.u32()):
        name = r.string()
        tag = r.u8()
        dims = [r.u32() for _ in range(r.u8())]
        count = int(np.prod(dims)) if dims else 1

        scale = r.f32() if tag == INT8 else 1.0
        itemsize = np.dtype(_NUMPY[tag]).itemsize
        flat = np.frombuffer(r.take(itemsize * count), dtype=_NUMPY[tag])
        tensors[name] = (flat.astype(np.float64) * scale).reshape(dims)

    return metadata, tensors
