import mmap
import torch
import ctypes
import os
import timeit

# : > preved.medved; shred -n 1 -s 90G preved.medved
path = "preved.medved"
with open(path, "r+") as f:
    mm = mmap.mmap(f.fileno(), 0)
cpu = torch.frombuffer(mm, dtype=torch.uint8)
t = timeit.Timer('gpu = cpu.cuda()', globals=globals())
print(t.timeit(number=1))
