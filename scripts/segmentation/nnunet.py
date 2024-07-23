import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import Compose, Lambda

from medical_diffusion.data.datasets import NiftiPair3ImageGenerator


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.relu(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.conv(x)
        x_down = self.pool(x)
        return x, x_down


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(UpBlock, self).__init__()
        # 注意：这里假设转置卷积的输出通道数等于输入通道数
        self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        # 在合并操作后，通道数将是 in_channels + skip_channels
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# 修改UNet3D类以正确实例化UpBlock
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128]):
        super(UNet3D, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Down part of UNet
        for feature in features:
            self.downs.append(DownBlock(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # Up part of UNet
        features = features[::-1]  # Reverse features to start from the bottleneck
        for idx in range(len(features) - 1):
            # 下一个特征，用于跳跃连接的特征融合
            skip_channels = features[idx + 1]
            self.ups.append(UpBlock(features[idx] * 2, features[idx], skip_channels))

        # 处理最后一个UpBlock，无需跳跃连接
        self.final_conv = nn.Conv3d(features[-1], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x, x_down = down(x)
            skip_connections.append(x)
            x = x_down

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(len(self.ups)):
            x = self.ups[idx](x, skip_connections[idx])

        x = self.final_conv(x)
        return self.sigmoid(x)



# Model initialization and example
model = UNet3D()
print(model)

pet_transform = Compose([
    Lambda(lambda t: torch.from_numpy(t).float()),  # 更高效的转换方式
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.permute(0, 3, 2, 1)),  # 调整维度顺序
])
ct_transform = Compose([
    Lambda(lambda t: torch.from_numpy(t).float()),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.permute(0,3,2, 1)),
])
tumor_transform = Compose([
    Lambda(lambda t: torch.from_numpy(t).float()),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.permute(0,3,2, 1)),
])
ds=NiftiPair3ImageGenerator("/home/zyl/working/202406_01/Task107_hecktor2021/labelsTrain","/home/zyl/working/202406_01/Task107_hecktor2021/imagesTrain","/home/zyl/working/202406_01/Task107_hecktor2021/imagesTrain",128,128,tumor_transform,ct_transform,pet_transform,combine_input=False)
print(len(ds),ds[0]['pet'].max(),ds[0]['pet'].min(),ds[0]['tumor'].max(),ds[0]['tumor'].min())

# 确保 UNet3D, ConvBlock, DownBlock, UpBlock 已经定义

# 数据加载器
batch_size = 2  # 根据您的GPU内存调整
train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8)

# 模型初始化
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = UNet3D(in_channels=1, out_channels=1).to(device)

# 优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# 训练过程
num_epochs = 50  # 根据需要调整
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        pet = batch['pet'].to(device)   # PET图像作为输入
        tumor = batch['tumor'].to(device)  # 肿瘤分割图作为标签

        # 前向传播
        optimizer.zero_grad()
        outputs = model(pet)

        # 计算损失
        loss = criterion(outputs, tumor)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 打印每个epoch的损失
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

print("Training complete.")
