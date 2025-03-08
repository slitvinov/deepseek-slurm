torchrun --nnodes 2 --node-rank 0 main.py --master-addr localhost &
torchrun --nnodes 2 --node-rank 1 main.py --master-addr localhost &
