master_addr=`srun --mpi=pmix sh -xeuc 'if test $SLURM_PROCID -eq 0; then hostname; fi'`

srun --mpi=pmix sh -xeuc '
python ~/.local/bin/torchrun --master-addr '$master_addr' --nnodes $SLURM_NTASKS --node-rank $SLURM_PROCID main.py'
