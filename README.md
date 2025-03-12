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

Set `TORCH_LOGS=+distributed,+dist_c10d` for more info.

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
$ cat out
rank=0 size=4 local_rank=0 node='holygpu8a27401.rc.fas.harvard.edu' bd10c331-fad8-00d2-4636-71a35767624d, [42, [1, 2, 3, 4]], 123.0
rank=1 size=4 local_rank=1 node='holygpu8a27401.rc.fas.harvard.edu' 197b3481-c4db-7cfd-b9b2-ca7fbb6ad66e, [42, [1, 2, 3, 4]], 123.0
rank=2 size=4 local_rank=0 node='holygpu8a29201.rc.fas.harvard.edu' bf87ac69-1c93-9a19-24f4-8179938fc4f9, [42, [1, 2, 3, 4]], 123.0
rank=3 size=4 local_rank=1 node='holygpu8a29201.rc.fas.harvard.edu' bdc0c1d7-cb4a-0f8c-b874-fcb0ba9ebdf3, [42, [1, 2, 3, 4]], 123.0
```
