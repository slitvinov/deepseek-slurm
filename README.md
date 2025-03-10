# Install PyTorch

```
salloc --gpus-per-node 1 -t 0-6 --mem=20Gb
```

with allocation

```
module load python/3.12.5-fasrc01
mamba create -n torch
. activate torch
(torch) $ mamba install -yq cuda-toolkit pytorch
```

