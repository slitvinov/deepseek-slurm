import struct
import sys
import json
import mmap
import math
import collections

import numpy as np

s2nd = {
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

s2size = {
    "BF16": 2,
    "BOOL": 1,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "I8": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
    "U8": 1,
}

size_per_type = collections.Counter()
var_per_type = collections.Counter()
var_per_shape = collections.Counter()
size_per_shape = collections.Counter()
for path in sys.argv[1:]:
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    n, = struct.unpack("Q", mm[:8])
    j = json.loads(mm[8:8 + n])
    for name, meta in j.items():
        if name != "__metadata__":
            dtype = meta['dtype']
            shape = meta['shape']
            offsets = meta['data_offsets']
            size = s2size[dtype] * math.prod(shape)
            # print(dtype, size, shape)
            size_per_type[dtype] += size
            var_per_type[dtype] += 1
            var_per_shape[tuple([dtype] + shape)] += 1
            size_per_shape[tuple([dtype] + shape)] += size
            '''
            x = np.ndarray(shape, s2nd[dtype], mm, offset=offsets[0] + n + 8)
            assert x.nbytes == offsets[1] - offsets[0]
            print(name, np.min(x), np.max(x), offsets[0], dtype)
            '''
print(f"{'type':10}{'number':>10}{'size':>14}")
for key, value in size_per_type.items():
    print(f"{key:10}{var_per_type[key]:10,}{value / (1 << 30):12.2f}Gb")

total_size = sum(size_per_type.values())
print()
print(f"{'type':10}{'shape':>15}{'number':>10}{'fraction':>10}")
for key, value in var_per_shape.items():
    dtype, *shape = key
    fraction = 100 * size_per_shape[key] / total_size
    if fraction > 0.01:
        s = f"{shape}"
        print(f"{dtype:10}{s:>15}{value:10,}{fraction:8.2f} %")
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
