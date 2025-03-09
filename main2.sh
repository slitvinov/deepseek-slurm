# salloc --gpus-per-node 2 -N 2 --mem 20Gb  --no-shell
# sh main2.sh --jobid 6022877
master_addr=`srun "$@" sh -xeuc 'if test $SLURM_PROCID -eq 0; then hostname; fi'`
srun "$@" sh -xeuc '
     torchrun \
       --master-addr '$master_addr' \
       --nnodes $SLURM_NNODES \
       --node-rank $SLURM_NODEID \
       --nproc-per-node gpu \
       main.py'
