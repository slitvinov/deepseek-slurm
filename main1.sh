TORCH_LOGS=+distributed,+dist_c10d PYTHONUNBUFFERED=1 mpiexec.openmpi -n 4 \
		sh -xeuc '
torchrun --nnodes $OMPI_COMM_WORLD_SIZE --node-rank $OMPI_COMM_WORLD_RANK main.py'
