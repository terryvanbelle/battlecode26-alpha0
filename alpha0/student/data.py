"""Read student-records-v1 files (gzip, big-endian fixed-size records)."""
import gzip
from pathlib import Path
import numpy as np
from .spec import HEADER, OBS, MASK_BYTES, RECORD

HEADER_DTYPE = np.dtype([('robot', '>i4'), ('round', '>i4'), ('label', '>i2'), ('team', 'i1'), ('king', 'i1'),
                         ('step', 'i1'), ('flags', 'i1'), ('reserved', '>i2')])


def read_records(path):
    raw = gzip.decompress(Path(path).read_bytes())
    if len(raw) % RECORD:
        raise ValueError(f'{path}: {len(raw)} bytes is not a multiple of record size {RECORD}')
    a = np.frombuffer(raw, np.uint8).reshape(-1, RECORD)
    head = a[:, :HEADER].copy().view(HEADER_DTYPE).reshape(-1)
    obs = a[:, HEADER:HEADER + OBS]
    mask = a[:, HEADER + OBS:]
    assert mask.shape[1] == MASK_BYTES
    return head, obs, mask


def unpack_mask(packed, n):
    return np.unpackbits(packed, axis=1, bitorder='little')[:, :n].astype(bool)
