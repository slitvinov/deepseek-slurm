mpiexec.openmpi -n 4 \
		sh -euc '
torchrun --nnodes $OMPI_COMM_WORLD_SIZE --node-rank $OMPI_COMM_WORLD_RANK main1.py'
