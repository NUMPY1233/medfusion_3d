import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from pathlib import Path
import nibabel as nib

from medical_diffusion.data.datasets import NiftiPair3ImageGenerator
from medical_diffusion.models.embedders.latent_embedders import VQVAE
from medical_diffusion.models.pipelines import DiffusionPipeline



def resize_img_4d_01(input_ndarray):
    input_tensor = torch.from_numpy(input_ndarray)
    input_tensor.unsqueeze_(0)
    input_numpy=input_tensor.numpy()
    c, h, w, d = input_numpy.shape
    scaled_img = np.where(input_numpy > 0.5, 1, 0)
    tumor_transform = Compose([
        Lambda(lambda t: torch.tensor(t).float()),
        Lambda(lambda t: t.transpose(3, 1)),
    ])
    output=tumor_transform(scaled_img)
    return output



path_out = Path.cwd() / 'results' / 'metrics' / 'nocrop_change'

pet_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.transpose(3, 1)),
])
ct_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.transpose(3, 1)),
])
tumor_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: t.transpose(3, 1)),
])
ds = NiftiPair3ImageGenerator("/home/zyl/working/202406_01/Task107_hecktor2021/labelsTrain",
                              "/home/zyl/working/202406_01/Task107_hecktor2021/imagesTrain",
                              "/home/zyl/working/202406_01/Task107_hecktor2021/imagesTrain", 128, 128, tumor_transform,
                              ct_transform, pet_transform,combine_input=False)
pet=ds[0]['pet']
print(pet.shape)
pet.unsqueeze_(0)

device = torch.device('cuda:1')
pet=pet.to(device)
model = VQVAE.load_from_checkpoint('/home/zyl/working/202406_01/scripts/runs/VQVAE/2024_06_19_121029/epoch=509-step=102000.ckpt')
model.to(device)

with torch.no_grad():
    result = model(pet)[0].clamp(-1, 1)
result = (result + 1) / 2  # Transform from [-1, 1] to [0, 1]
results = result.clamp(0, 1)
path_out = Path(path_out)
path_out.mkdir(parents=True, exist_ok=True)

sample = result.squeeze(0).squeeze(0).detach().cpu().numpy()
nifti_img_s = nib.Nifti1Image(sample, affine=np.eye(4))
nib.save(nifti_img_s, '/home/zyl/working/202406_01/scripts/runs/results/metrics/nocrop_change/sample10.nii.gz')

