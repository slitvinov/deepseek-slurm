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

for i in range(size):
    if rank == i:
        print(rank, size, node, *objects)
    dist.barrier()
dist.destroy_process_group()
