for i in 100Mb 500Mb 1Gb 2Gb 4Gb 8Gb 16Gb 32Gb 64Gb 90Gb
do : > preved.medved; shred -n 1 -s $i file.raw
   echo $i
   python3 -c '
import mmap
import torch
import timeit

path = "preved.medved"
with open(path, "r+") as f:
    mm = mmap.mmap(f.fileno(), 0)
cpu = torch.frombuffer(mm, dtype=torch.uint8)
t = timeit.Timer("gpu = cpu.cuda()", globals=globals())
print(f"{8 * len(mm) / (1 << 10 << 10 << 10) / t.timeit(number=1):.2f} GBit/second")
'
done | xargs -n 2
