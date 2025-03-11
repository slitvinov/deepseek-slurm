import torch
import torch.distributed as dist
import torch.distributed.elastic.multiprocessing.errors
import datetime
import platform


@dist.elastic.multiprocessing.errors.record
def main():
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=30))
    local_rank = dist.get_node_local_rank()
    node = platform.node()
    rank = dist.get_rank()
    size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    if rank == 0:
        tensor = torch.tensor(123, dtype=torch.bfloat16, device="cuda")
        objects = [42, [1, 2, 3, 4]]
    else:
        tensor = torch.empty([], dtype=torch.bfloat16, device="cuda")
        objects = [None, None]
    dist.broadcast_object_list(objects, 0)
    dist.broadcast(tensor, 0)

    for i in range(size):
        if rank == i:
            print(
                f"{rank=} {size=} {local_rank=} {node=} "
                f"{torch.cuda.get_device_properties().uuid}, {objects}, {tensor}"
            )
        dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
