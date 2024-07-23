import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchio import Lambda
from torchvision.transforms import Compose

from medical_diffusion.data.datasets import NiftiPair3ImageGenerator




pet_transform = Compose([
    Lambda(lambda t: torch.from_numpy(t).float() if isinstance(t, np.ndarray) else t),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.transpose(3, 1)),
])
ct_transform = Compose([
    Lambda(lambda t: torch.from_numpy(t).float() if isinstance(t, np.ndarray) else t),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.transpose(3, 1)),
])
tumor_transform = Compose([
    Lambda(lambda t: torch.from_numpy(t).float() if isinstance(t, np.ndarray) else t),
    Lambda(lambda t: t.transpose(3, 1)),
])
ds=NiftiPair3ImageGenerator("/home/zyl/working/202406_01/Task107_hecktor2021/labelsTrain","/home/zyl/working/202406_01/Task107_hecktor2021/imagesTrain","/home/zyl/working/202406_01/Task107_hecktor2021/imagesTrain",128,128,tumor_transform,ct_transform,pet_transform,combine_input=False)


# 实验不同的 num_workers 值
best_num_workers = 0
best_time = float('inf')

for num_workers in range(1, 49):  # 从 1 到 48 进行尝试
    dataloader = DataLoader(ds, batch_size=4, num_workers=num_workers)
    start_time = time.time()
    for batch in dataloader:
        pass
    total_time = time.time() - start_time
    print(f'num_workers={num_workers}, time={total_time:.4f} seconds')
    if total_time < best_time:
        best_time = total_time
        best_num_workers = num_workers

print(f'Best num_workers: {best_num_workers}, Best time: {best_time:.4f} seconds')
