import torch
import torch.distributed as dist
import datetime
import platform

dist.init_process_group(timeout=datetime.timedelta(seconds=10))
node = platform.node()
rank = dist.get_rank()
size = dist.get_world_size()

if rank == 0:
    objects = [42, [1, 2, 3, 4]]
    dist.broadcast_object_list(objects, 0)
else:
    objects = [None, None]
    dist.broadcast_object_list(objects, 0)

if rank == 0:
    x = torch.tensor(123, dtype=float)
    dist.send(x, 1)
elif rank == 1:
    x = torch.empty(1, dtype=float)
    dist.recv(x, 0)
else:
    x = None

for i in range(size):
    if rank == i:
        print(rank, size, node, x, *objects)
    dist.barrier()
dist.destroy_process_group()
