## Install PyTorch

```
module load python/3.12.5-fasrc01
mamba create -n torch
sbatch --gpus-per-node 1 -t 0-6 --mem 10Gb --wrap 'srun mamba install -n torch -yq cuda-toolkit pytorch'
```

## Check GPUs

```
$ module load python/3.12.5-fasrc01
$ sbatch -p seas_gpu --gpus-per-node 2 --mem 20Gb -o out -e err --wrap '
srun mamba run --no-capture-output -n torch python <<'\''!'\''
import torch
for i in range(torch.cuda.device_count()):
   p = torch.cuda.get_device_properties(i)
   print(f"{p.name}; {p.total_memory >> 30}Gb; {p.uuid}")
!
'
$ cat out
NVIDIA A100-SXM4-80GB; 79Gb; f10d781b-9819-eaa3-badb-a55dd1398607
NVIDIA A100-SXM4-80GB; 79Gb; fbf548fc-e161-1853-0d59-2c12f84acf7e
```

## Check connectivity

```
$ module load python/3.12.5-fasrc01
$ sbatch -p seas_gpu --gpus-per-node 2 -N 2 --mem 20Gb -o out -e err --wrap '
set -uex -o pipefail
master_addr=`srun -n 1 -N 1 hostname`
PYTHONUNBUFFERED=1 srun -u sh -c '\''mamba run --no-capture-output -n torch torchrun \
       --rdzv-backend static \
       --master-addr "'\''$master_addr'\''" \
       --nnodes $SLURM_NNODES \
       --node-rank $SLURM_NODEID \
       --nproc-per-node gpu \
       main1.py'\''
'
```
