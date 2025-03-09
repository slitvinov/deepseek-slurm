import struct
import sys

path = sys.argv[1]
with open(path, "rb") as f:
    fmt = "Q"
    n, = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
    print(f.read(n))
