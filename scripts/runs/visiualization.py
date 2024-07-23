import pandas as pd
import matplotlib.pyplot as plt
path='/home/zyl/working/202406_01/scripts/runs/LDM_VQGAN2/2024_06_19_130913/lightning_logs/version_0/metrics.csv'
# 读取 CSV 文件
vlisty=[ 'train/loss_epoch', 'train/emb_loss_epoch', 'train/L2_epoch', 'train/L1_epoch', 'train/ssim_epoch', 'train/loss_step', 'train/emb_loss_step', 'train/L2_step', 'train/L1_step', 'train/ssim_step']
vlistx=['epoch', 'step']
dlisty=['train/loss_epoch','train/L2_epoch', 'train/L1_epoch','train/loss_step','train/L2_step', 'train/L1_step', ]
dlistx=['epoch', 'step']
df = pd.read_csv(path)
fig,axes=plt.subplots(2,3, figsize=(30, 10))
for i,ax in enumerate(axes.flat):
    ax.plot(df[dlistx[i//3]],df[dlisty[i]],label=dlisty[i])
plt.tight_layout()
plt.legend()
plt.show()