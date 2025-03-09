mpiexec.openmpi -n 4 \
		sh -xeuc '
PYTHONUNBUFFERED=1 torchrun --nnodes $OMPI_COMM_WORLD_SIZE --node-rank $OMPI_COMM_WORLD_RANK main.py'
