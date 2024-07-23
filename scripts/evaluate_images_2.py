from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights
from medical_diffusion.data.datasets import SimpleDataset3D
from scipy.linalg import sqrtm
from pytorch_msssim import ms_ssim


from sklearn.metrics import mean_squared_error

def calculate_mse(features_real, features_fake):
    # 确保两个特征集的形状相同
    mse = mean_squared_error(features_real, features_fake)
    return mse

from torch_two_sample import MMDStatistic

def calculate_mmd(features_real, features_fake):
    # 将 numpy 数组转换为 torch tensor
    real = torch.tensor(features_real, device=device, dtype=torch.float32)
    fake = torch.tensor(features_fake, device=device, dtype=torch.float32)
    # 计算 MMD
    mmd_stat = MMDStatistic(real.size(0), fake.size(0))
    mmd_value = mmd_stat(real, fake, alphas=[0.5], ret_matrix=False)
    return mmd_value.item()


def calculate_ms_ssim(real_images, fake_images):
    # 将 numpy 数组转换为 torch tensor，并确保它们在正确的设备上
    real_images = torch.tensor(real_images, device=device, dtype=torch.float32)
    fake_images = torch.tensor(fake_images, device=device, dtype=torch.float32)
    # 计算 MS-SSIM
    msssim = ms_ssim(real_images, fake_images, data_range=1.0, size_average=True)  # 确保数据范围正确
    return msssim.item()


# ----------------Settings --------------
batch_size = 1
path_out = Path.cwd()/'results'/'pet'/'metrics'
path_out.mkdir(parents=True, exist_ok=True)

# ----------------- Logging -----------
current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger.addHandler(logging.FileHandler(path_out/f'metrics_{current_time}.log', 'w'))

# -------------- Helpers ---------------------
class FeatureExtractor3D(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor3D, self).__init__()
        self.model = model
        self.model.fc = nn.Identity()  # Remove the final classification layer

    def forward(self, x):
        return self.model(x)

def extract_features_3d(model, dataloader, device):
    model.eval()
    features = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            imgs = batch['source'].to(device)
            imgs = imgs.repeat(1, 3, 1, 1, 1)  # 将单通道扩展为3通道
            feats = model(imgs).cpu().numpy()
            features.append(feats)
    features = np.concatenate(features, axis=0)
    return features

def calculate_fid(mu1, sigma1, mu2, sigma2):
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrtm
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
def calculate_pr(features_real, features_fake, k=1):
    from sklearn.metrics import pairwise_distances

    # 计算所有真实特征和假特征之间的距离
    dists = pairwise_distances(features_real, features_fake, metric='euclidean')

    # 计算召回率: 至少有 k 个生成特征在真实特征的某个距离阈值范围内
    recall = (np.sum(np.min(dists, axis=1) <= k) / features_real.shape[0])

    # 计算精确度: 至少有 k 个真实特征在生成特征的某个距离阈值范围内
    precision = (np.sum(np.min(dists, axis=0) <= k) / features_fake.shape[0])

    return precision, recall

# ---------------- Dataset/Dataloader ----------------
path = '/home/zyl/working/202406_01/scripts/runs/results/metrics/nocrop'
ds_real = SimpleDataset3D(path, keyword='target')
ds_fake = SimpleDataset3D(path, keyword='sample')

dm_real = DataLoader(ds_real, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)
dm_fake = DataLoader(ds_fake, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)

logger.info(f"Samples Real: {len(ds_real)}")
logger.info(f"Samples Fake: {len(ds_fake)}")

# ------------- Init Metrics ----------------------
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights)
feature_extractor = FeatureExtractor3D(model).to(device)

# --------------- Start Calculation -----------------
features_real = extract_features_3d(feature_extractor, dm_real, device)
features_fake = extract_features_3d(feature_extractor, dm_fake, device)



# 在提取特征后计算精确度和召回率
precision, recall = calculate_pr(features_real, features_fake, k=6)  # 这里的 k 值可以根据需要调整

# 记录精确度和召回率
logger.info(f"Precision: {precision}")
logger.info(f"Recall: {recall}")



# -------------- Compute FID -------------------
mu_real = np.mean(features_real, axis=0)
sigma_real = np.cov(features_real, rowvar=False)
mu_fake = np.mean(features_fake, axis=0)
sigma_fake = np.cov(features_fake, rowvar=False)

fid_value = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
logger.info(f"3D FID Score: {fid_value}")



