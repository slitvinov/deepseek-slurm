## Install PyTorch

```
module load python/3.12.5-fasrc01
mamba create -n torch
sbatch --gpus-per-node 1 -t 0-6 --mem 10Gb --wrap 'mamba install -n torch -yq cuda-toolkit pytorch'
```

## Check GPUs

In batch
```
module load python/3.12.5-fasrc01
srun -p seas_gpu --gpus-per-node 2 --mem 20Gb sh -c '
mamba run --no-capture-output -n env python <<!
import torch
if torch.cuda.is_available():
   for i in range(torch.cuda.device_count()):
      p = torch.cuda.get_device_properties(i)
      print(f"{p.name}; {p.total_memory >> 30}Gb; {p.uuid}")
else:
   print("CUDA is not available")
!
'
srun: job 6169362 queued and waiting for resources
srun: job 6169362 has been allocated resources
NVIDIA H100 80GB HBM3; 79Gb; 6e24bbf3-2078-034a-e506-785bb3c21cbf
NVIDIA H100 80GB HBM3; 79Gb; 7b12d7a7-052f-8086-af3d-e89744b174c4
```

Interactivly
```
module load python/3.12.5-fasrc01
. activate torch
salloc -p seas_gpu --gpus-per-node 2 --mem 20Gb
python <<!
import torch
if torch.cuda.is_available():
   for i in range(torch.cuda.device_count()):
      p = torch.cuda.get_device_properties(i)
      print(f"{p.name}; {p.total_memory >> 30}Gb; {p.uuid}")
else:
   print("CUDA is not available")
!
```
