from pathlib import Path
import torch
from torchvision import utils
import math

from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.models.pipelines import DiffusionPipeline
import logging
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from medical_diffusion.data.datasets import NiftiPairImageGenerator, NiftiPair3ImageGenerator
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import nibabel as nib

torch.manual_seed(0)
device = torch.device('cuda:2')

# ------------ Load Model ------------
# pipeline = DiffusionPipeline.load_best_checkpoint(path_run_dir)
# pipeline = DiffusionPipeline.load_from_checkpoint(
#     "/home/zyl/working/202406_01/scripts/runs/LDM_VQGAN2/2024_06_19_130913/epoch=1529-step=153000.ckpt")
pipeline = DiffusionPipeline.load_from_checkpoint(
    "/home/zyl/working/202406_01/scripts/runs/LDM_VQVAE/2024_06_19_122405/epoch=1829-step=183000.ckpt")
pipeline.to(device)
tumorfolder='/home/zyl/working/202406_01/Task107_hecktor2021/labelsTest/'
ctfolder='/home/zyl/working/202406_01/Task107_hecktor2021/imagesTest/'
petfolder='/home/zyl/working/202406_01/Task107_hecktor2021/imagesTest/'

input_size = 128
depth_size = 128
with_condition = True

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

# ----------------Settings --------------
batch_size = 1
max_samples = None  # set to None for all
target_class = None  # None for no specific class
# path_out = Path.cwd()/'results'/'MSIvsMSS_2'/'metrics'
# path_out = Path.cwd()/'results'/'AIROGS'/'metrics'
path_out = Path.cwd() / 'results' / 'metrics' / 'nocrop'
path_out.mkdir(parents=True, exist_ok=True)
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

# ----------------- Logging -----------
current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger.addHandler(logging.FileHandler(path_out / f'metrics_{current_time}.log', 'w'))

# ---------------- Dataset/Dataloader ----------------
dataset = NiftiPair3ImageGenerator(
    tumorfolder,
    ctfolder,
    petfolder,
    input_size=input_size,
    depth_size=depth_size,
    tumor_transform=tumor_transform,
    ct_transform=ct_transform,
    pet_transform=pet_transform,
    full_channel_mask=True,
    combine_input=False
)


dl= DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


# --------- Generate Samples  -------------------
steps = 250
use_ddim = True
images = {}
n_samples = 1

for i, batch in enumerate(dl):
    torch.manual_seed(0)

    x_0 = batch['pet'].to(device)
    # condition = torch.cat((batch['tumor'],batch['ct']),dim=1).to(device)
    condition = batch['tumor'].to(device)
    print(x_0.max())
    print(x_0.min())
    x_0 = (x_0 + 1) / 2
    print(x_0.max())
    print(x_0.min())
    target_img1 = x_0.squeeze(0).squeeze(0).detach().cpu().numpy()
    nifti_img_t = nib.Nifti1Image(target_img1, affine=np.eye(4))
    nib.save(nifti_img_t, path_out / f'target_{i}.nii.gz')
    # --------- Conditioning ---------
    # un_cond = torch.tensor([1-cond]*n_samples, device=device)
    un_cond = None

    # ----------- Run --------
    results = pipeline.sample(n_samples, (4, 16, 16, 16), condition=condition, guidance_scale=1, steps=steps,
                              use_ddim=use_ddim)

    # --------- Save result ---------------
    results = (results + 1) / 2  # Transform from [-1, 1] to [0, 1]
    results = results.clamp(0, 1)

    path_out = Path(path_out)
    path_out.mkdir(parents=True, exist_ok=True)

    sample_img1 = results.squeeze(0).squeeze(0).detach().cpu().numpy()
    nifti_img_s = nib.Nifti1Image(sample_img1, affine=np.eye(4))
    nib.save(nifti_img_s, path_out / f'sample_{i}.nii.gz')

    tumor_img1 = batch['tumor'].squeeze(0).squeeze(0).detach().cpu().numpy()
    nifti_img_tumor = nib.Nifti1Image(tumor_img1, affine=np.eye(4))
    nib.save(nifti_img_tumor, path_out / f'tumor_{i}.nii.gz')

    ct_img1 = batch['ct']
    ct_img1=(ct_img1+1)/2
    ct_img1 = ct_img1.squeeze(0).squeeze(0).detach().cpu().numpy()
    nifti_img_ct = nib.Nifti1Image(ct_img1, affine=np.eye(4))
    nib.save(nifti_img_ct, path_out / f'ct_{i}.nii.gz')


