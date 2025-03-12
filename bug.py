import torch
import torch.distributed as dist
import torch.distributed.elastic.multiprocessing.errors

@dist.elastic.multiprocessing.errors.record
def main():
    dist.init_process_group()
    rank = dist.get_rank()
    size = dist.get_world_size()
    x = torch.tensor(0)
    if rank == 0:
        x = torch.tensor(123)
        dist.send(x, 1)
    elif rank == 1:
        dist.recv(x, 0)
    dist.barrier()
    for i in range(size):
        if rank == i:
            print(f"{rank=} {size=}  {x=}")
        dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
