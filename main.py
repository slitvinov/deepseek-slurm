import torch
import datetime
import platform
import torch.distributed as dist
import torch.distributed.elastic.multiprocessing.errors


@dist.elastic.multiprocessing.errors.record
def main():
    dist.init_process_group(timeout=datetime.timedelta(seconds=10))
    node = platform.node()
    rank = dist.get_rank()
    size = dist.get_world_size()

    if rank == 0:
        objects = [42, [1, 2, 3, 4]]
    else:
        objects = [None, None]
    dist.broadcast_object_list(objects, 0)

    if rank == 0:
        x = torch.tensor(123, dtype=float)
        dist.send(x, 1)
    elif rank == 1:
        x = torch.empty([], dtype=float)
        dist.recv(x, 0)
    else:
        x = None

    for i in range(size):
        if rank == i:
            print(rank, size, node, x, *objects)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
