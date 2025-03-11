#!/bin/sh

# module load python cuda
# mamba activate env
# sbatch -p seas_gpu -N 4 --gpus-per-node 4 --mem 0 --constrain h100 -t 0-2 -J 4x4 run.sh
# or
# salloc -N 4 --gpus-per-node 4 --mem 20Gb  --no-shell
# sh run.sh --jobid 6022877
master_addr=`srun "$@" sh -xeuc 'if test $SLURM_PROCID -eq 0; then hostname; fi'`
srun -l -u "$@" sh -xeuc '
     PYTHONUNBUFFERED=1 torchrun \
       --master-addr '$master_addr' \
       --nnodes $SLURM_NNODES \
       --node-rank $SLURM_NODEID \
       --nproc-per-node gpu \
       --rdzv-backend static \
	   DeepSeek-V3/inference/generate.py \
	   --ckpt-path weights.split \
	   --config DeepSeek-V3/inference/configs/config_671B.json \
	   --input-file input'
