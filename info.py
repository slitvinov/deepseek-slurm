import struct
import sys
import json
import mmap

import numpy as np

s2n = {
    "BOOL": bool,
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "I8": np.int8,
    "U16": np.uint16,
    "U32": np.uint32,
    "U64": np.uint64,
    "U8": np.uint8,
}

path = sys.argv[1]
with open(path, "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

n, = struct.unpack("Q", mm[:8])
j = json.loads(mm[8:8 + n])
for name, meta in j.items():
    if name != "__metadata__":
        dtype = s2n[meta['dtype']]
        shape = meta['shape']
        offsets = meta['data_offsets']
        x = np.ndarray(shape, dtype, mm, offset=offsets[0] + n + 8)
        assert x.nbytes == offsets[1] - offsets[0]
        print(name, np.min(x), np.max(x), offsets[0], dtype)
        # print(name, dtype, shape, offsets[0])

'''
from safetensors.torch import save
import torch
tensors = {
    "embedding": torch.zeros((512, 1024)),
    "attention": torch.zeros((256, 256), dtype=torch.uint8)
}
with open("info.safetensors", "wb") as f:
    f.write(save(tensors))
'''
