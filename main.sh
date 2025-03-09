OMP_NUM_THREADS=1
export OMP_NUM_THREADS
torchrun --nnodes 2 --node-rank 0 --nproc-per-node cpu main.py &
torchrun --nnodes 2 --node-rank 1 --nproc-per-node cpu main.py &
