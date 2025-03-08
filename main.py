import torch
import torch.distributed as dist
import datetime

dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=10))
rank = dist.get_rank()
size = dist.get_world_size()

if rank == 0:
    objects = [42]
    dist.broadcast_object_list(objects, 0)
else:
    objects = [None]
    dist.broadcast_object_list(objects, 0)

for i in range(size):
    if rank == i:
        print(rank, size, *objects)
    dist.barrier()
dist.destroy_process_group()
