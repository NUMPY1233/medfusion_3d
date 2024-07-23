import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda
import torch

from medical_diffusion.data.datasets import NiftiPair3ImageGenerator

def convert_to_tensor(x):
    # 检查输入是否为Tensor，如果不是，则从NumPy数组转换
    if not torch.is_tensor(x):
        x = torch.from_numpy(x)
    x = x.float()

    x = (x * 2) - 1

    if x.dim() == 4 and x.shape[0] not in [1, 3]:  # 假设通道数只能是1或3
        x = x.permute(3, 0, 1, 2)

    return x
# pet_transform = Compose([
#     Lambda(convert_to_tensor)
# ])
#
# ct_transform = Compose([
#     Lambda(convert_to_tensor)
# ])
#
# tumor_transform = Compose([
#     Lambda(convert_to_tensor)
# ])

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
print(len(ds))
test=ds[0]
pet=test['pet']
ct=test['ct']
tumor=test['tumor']
print(pet.shape,ct.shape,tumor.shape)
print(pet.max().dtype,pet.min(),ct.max().dtype,ct.min(),tumor.max().dtype,tumor.min())
# dataloader = DataLoader(ds, batch_size=10, num_workers=8,shuffle=True)