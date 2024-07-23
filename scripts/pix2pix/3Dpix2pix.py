import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

from torchio import Lambda
from torchvision.transforms import transforms, Compose


from medical_diffusion.data.datasets import NiftiPair3ImageGenerator
from scripts.pix2pix.models import Discriminator3D, Generator3D


def save_checkpoint(epoch, generator, discriminator, g_optimizer, d_optimizer, path):
    state = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
    }
    torch.save(state, path)

def train(dataloader, generator, discriminator, g_optimizer, d_optimizer, criterion, device):
    save_dir = '/home/zyl/working/202406_01/scripts/runs/pix2pix'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_epochs = 200  # 训练周期数可以根据需要调整
    save_interval = 10  # 保存间隔，可以根据需要调整

    for epoch in range(num_epochs):  # 使用num_epochs替代硬编码的200
        for i, data_batch in enumerate(dataloader):
            masks = data_batch['tumor'].to(device).float()
            pets = data_batch['pet'].to(device).float()
            b_size = masks.size(0)

            # Real and fake labels setup
            real_labels = torch.full((b_size, 1, 16, 16, 16), 1.0, device=device, dtype=torch.float)
            fake_labels = torch.full((b_size, 1, 16, 16, 16), 0.0, device=device, dtype=torch.float)

            # Train Discriminator
            discriminator.zero_grad()
            real_input = torch.cat([masks, pets], dim=1)
            real_output = discriminator(real_input)
            d_loss_real = criterion(real_output, real_labels)

            fake_pets = generator(masks)
            fake_input = torch.cat([masks, fake_pets], dim=1)
            fake_output = discriminator(fake_input)
            d_loss_fake = criterion(fake_output, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward(retain_graph=True)
            d_optimizer.step()

            # Train Generator
            generator.zero_grad()
            reprocessed_fake_output = discriminator(fake_input)
            g_loss = criterion(reprocessed_fake_output, real_labels)
            g_loss.backward()
            g_optimizer.step()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

        # 每个 epoch 结束时保存模型
        if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
            save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            save_checkpoint(epoch + 1, generator, discriminator, g_optimizer, d_optimizer, path=save_path)

# 示例调用
# train(dataloader, generator, discriminator, g_optimizer, d_optimizer, criterion, device)



# 初始化网络和优化器
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
generator = Generator3D().to(device)
discriminator = Discriminator3D().to(device)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# 数据加载
from torchvision.transforms import Compose
import torch

pet_transform = Compose([
    Lambda(lambda t: torch.from_numpy(t).float() if isinstance(t, np.ndarray) else t),
    # Lambda(lambda t: (t * 2) - 1), #保持在0-1即可，无需变到-1-+1
    Lambda(lambda t: t.transpose(3, 1)),
])
ct_transform = Compose([
    Lambda(lambda t: torch.from_numpy(t).float() if isinstance(t, np.ndarray) else t),
    # Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.transpose(3, 1)),
])
tumor_transform = Compose([
    Lambda(lambda t: torch.from_numpy(t).float() if isinstance(t, np.ndarray) else t),
    # Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.transpose(3, 1)),
])


ds=NiftiPair3ImageGenerator("/home/zyl/working/202406_01/Task107_hecktor2021/labelsTrain","/home/zyl/working/202406_01/Task107_hecktor2021/imagesTrain","/home/zyl/working/202406_01/Task107_hecktor2021/imagesTrain",128,128,tumor_transform,ct_transform,pet_transform,combine_input=False)
dataloader = DataLoader(ds, batch_size=10, num_workers=20,shuffle=True,pin_memory=True)

# 开始训练
train(dataloader, generator, discriminator, g_optimizer, d_optimizer, criterion, device)