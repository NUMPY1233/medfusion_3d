from pathlib import Path
import torch 
from torchvision import utils 
import math 
from medical_diffusion.models.pipelines import DiffusionPipeline
import logging
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from medical_diffusion.data.datasets import NiftiPairImageGenerator
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import nibabel as nib
torch.manual_seed(0)
device = torch.device('cuda')

# ------------ Load Model ------------
# pipeline = DiffusionPipeline.load_best_checkpoint(path_run_dir)
pipeline = DiffusionPipeline.load_from_checkpoint("/home/zyl/working/202406_01/scripts/runs/LDM_VQVAE/2024_06_19_122405/epoch=2039-step=204000.ckpt")
pipeline.to(device)

inputfolder = "/home/zyl/working/202406_01/Task107_hecktor2021/labelsTest/"
targetfolder = "/home/zyl/working/202406_01/Task107_hecktor2021/imagesTest/"
input_size = 128
depth_size = 128
with_condition =  True

transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.transpose(3, 1)),
])

input_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    # Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.transpose(3, 1)),
])

# ----------------Settings --------------
batch_size = 1
max_samples = None # set to None for all 
target_class = None # None for no specific class 
# path_out = Path.cwd()/'results'/'MSIvsMSS_2'/'metrics'
# path_out = Path.cwd()/'results'/'AIROGS'/'metrics'
path_out = Path.cwd()/'results'/'metrics'/ 'nocrop'
path_out.mkdir(parents=True, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------- Logging -----------
current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger.addHandler(logging.FileHandler(path_out/f'metrics_{current_time}.log', 'w'))

# ---------------- Dataset/Dataloader ----------------
dataset = NiftiPairImageGenerator(
    inputfolder,
    targetfolder,
    input_size=input_size,
    depth_size=depth_size,
    transform=input_transform if with_condition else transform,
    target_transform=transform,
    full_channel_mask=True
)


dl = DataLoader(dataset, batch_size = 1, shuffle=False, num_workers=1, pin_memory=True)

# --------- Generate Samples  -------------------
steps = 250
use_ddim = True 
images = {}
n_samples = 1

for i,batch in enumerate(dl):
# batch = next(iter(dl))
    torch.manual_seed(0)
    x_0 = batch['target']
    condition = batch['input'].cuda()

    print(x_0.max())
    print(x_0.min())
    x_0 = (x_0 + 1) / 2
    print(x_0.max())
    print(x_0.min())
    target_img1 = x_0.squeeze(0).squeeze(0).detach().cpu().numpy()
    nifti_img_t = nib.Nifti1Image(target_img1, affine = np.eye(4))
    nib.save(nifti_img_t, path_out/f'target_{i}.nii.gz')  
    # --------- Conditioning ---------
    # un_cond = torch.tensor([1-cond]*n_samples, device=device)
    un_cond = None 

    # ----------- Run --------
    results = pipeline.sample(n_samples, (4, 16, 16, 16), condition=condition, guidance_scale=1,  steps=steps, use_ddim=use_ddim )
    # results = pipeline.sample(n_samples, (4, 64, 64), guidance_scale=1, condition=condition, un_cond=un_cond, steps=steps, use_ddim=use_ddim )

    # --------- Save result ---------------
    results = (results+1)/2  # Transform from [-1, 1] to [0, 1]
    results = results.clamp(0, 1)


    path_out = Path(path_out)
    path_out.mkdir(parents=True, exist_ok=True)

    sample_img1 = results.squeeze(0).squeeze(0).detach().cpu().numpy()
    
    nifti_img_s = nib.Nifti1Image(sample_img1, affine = np.eye(4))

    nib.save(nifti_img_s, path_out/f'sample_{i}.nii.gz')  
    condition_img1 = condition.squeeze(0).squeeze(0).detach().cpu().numpy()
    nifti_img_c = nib.Nifti1Image(condition_img1, affine = np.eye(4))
    nib.save(nifti_img_c, path_out/f'condition_{i}.nii.gz')

    img = x_0[0, 0,:,:,:]
    fake = results[0, 0,:,:,:]

    img = img.cpu().numpy()
    fake = fake.cpu().numpy()
    fig, axs = plt.subplots(nrows=1, ncols=3)
    for ax in axs:
        ax.axis("off")
    ax = axs[0]
    ax.imshow(img[..., img.shape[2] // 2], cmap="gray")
    ax = axs[1]
    ax.imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
    ax = axs[2]
    ax.imshow(img[img.shape[0] // 2, ...], cmap="gray")

    fig, axs = plt.subplots(nrows=1, ncols=3)
    for ax in axs:
        ax.axis("off")
    ax = axs[0]
    ax.imshow(fake[..., fake.shape[2] // 2], cmap="gray")
    ax = axs[1]
    ax.imshow(fake[:, fake.shape[1] // 2, ...], cmap="gray")
    ax = axs[2]
    ax.imshow(fake[fake.shape[0] // 2, ...], cmap="gray")