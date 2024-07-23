from torch import nn
import torch
from torchvision import transforms


class Generator3D(nn.Module):
    def __init__(self):
        super(Generator3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1),  # 64x64x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),  # 32x32x32
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),  # 16x16x16
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),  # 32x32x32
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),  # 64x64x64
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1),  # 128x128x128
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Discriminator3D(nn.Module):
    def __init__(self):
        super(Discriminator3D, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=4, stride=2, padding=1),  # 64x64x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),  # 32x32x32
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 1, kernel_size=4, stride=2, padding=1),  # 16x16x16
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)



if __name__=="__main__":
    input=torch.randn((4,2,128,128,128))
    model=Discriminator3D()
    output=model(input)
    print(input.shape)
    print(output.shape)
