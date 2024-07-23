
import torch.utils.data as data 
from pathlib import Path 
from torchvision import transforms as T

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from glob import glob
import matplotlib.pyplot as plt
import nibabel as nib
import torchio as tio
import numpy as np
import torch
import re
import os
import scipy.ndimage as snd
import torchio as tio 

from medical_diffusion.data.augmentation.augmentations_3d import ImageToTensor


from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda



class SimpleDataset3D(data.Dataset):
    def __init__(
        self,
        path_root,
        item_pointers =[],
        # crawler_ext = ['nii.gz'], # other options are ['nii.gz'],
        keyword='sample',#other options are 'target'
        transform = None,
        image_resize = None,
        flip = False,
        image_crop = None,
        use_znorm=True, # Use z-Norm for MRI as scale is arbitrary, otherwise scale intensity to [-1, 1]
    ):
        super().__init__()
        self.path_root = path_root
        self.keyword=keyword
        # self.crawler_ext = crawler_ext

        if transform is None: 
            self.transform = T.Compose([
                tio.Resize(image_resize) if image_resize is not None else tio.Lambda(lambda x: x),
                tio.RandomFlip((0,1,2)) if flip else tio.Lambda(lambda x: x),
                tio.CropOrPad(image_crop) if image_crop is not None else tio.Lambda(lambda x: x),
                tio.ZNormalization() if use_znorm else tio.RescaleIntensity((-1,1)),
                ImageToTensor() # [C, W, H, D] -> [C, D, H, W]
            ])
        else:
            self.transform = transform


    def __len__(self):
        return len(self.get_file_names())
    def get_file_names(self):
        # files = sorted(glob(os.path.join(self.path_root, '*_0000.nii.gz')))
        files = sorted(glob(os.path.join(self.path_root, self.keyword+'_*.nii.gz')))
        return files
    def __getitem__(self, index):
        file_names=self.get_file_names()
        file_path= file_names[index]
        # path_item = self.path_root/rel_path_item
        img = self.load_item(file_path)
        return {'uid':file_path, 'source': self.transform(img)}
    
    def load_item(self, path_item):
        return tio.ScalarImage(path_item) # Consider to use this or tio.ScalarLabel over SimpleITK (sitk.ReadImage(str(path_item)))
    
    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        return [path.relative_to(path_root) for path in Path(path_root).rglob(f'*.{extension}')]
class NiftiPairImageGenerator(Dataset):
    def __init__(self,
            input_folder: str,
            target_folder: str,
            input_size: int,
            depth_size: int,
            input_channel: int = 2,
            transform=None,
            target_transform=None,
            full_channel_mask=False,
            combine_output=False
        ):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.pair_files = self.pair_file()
        self.input_size = input_size
        self.depth_size = depth_size
        self.input_channel = input_channel
        self.scaler = MinMaxScaler()
        self.transform = transform
        self.target_transform = target_transform
        self.full_channel_mask = full_channel_mask
        self.combine_output = combine_output

    def pair_file(self):
        input_files = sorted(glob(os.path.join(self.input_folder, '*')))
        target_files = sorted(glob(os.path.join(self.target_folder, '*_0000.nii.gz')))
        pairs = []
        for input_file, target_file in zip(input_files, target_files):
            assert int("".join(re.findall("\d", input_file))) == int("".join(re.findall("\d", target_file))[:-4])
            pairs.append((input_file, target_file))
        return pairs

    def read_image(self, file_path, pass_scaler=False):
        img = nib.load(file_path).get_fdata()
        img = img.clip(min = 0)
        img = np.expand_dims(img,0)
        return img

    def plot(self, index, n_slice=30):
        data = self[index]
        input_img = data['input']
        target_img = data['target']
        plt.subplot(1, 2, 1)
        plt.imshow(input_img[:, :, n_slice])
        plt.subplot(1, 2, 2)
        plt.imshow(target_img[:, :, n_slice])
        plt.show()

    def resize_img(self, img):
        h, w, d = img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(tensor=img[np.newaxis, ...])
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(cop(img))[0]
        return img

    def resize_img_4d(self, input_img):
        c, h, w, d = input_img.shape
        scaled_img = snd.zoom(input_img, [c, self.input_size / h, self.input_size / w, self.depth_size / d])
        return scaled_img.clip(min = 0)

    def resize_img_4d_pad(self, input_img):
        c, h, w, d = input_img.shape
        pad_one_side = (self.input_size - h) // 2
        padding = [(0,0), (pad_one_side,pad_one_side), (pad_one_side,pad_one_side), (pad_one_side,pad_one_side)]
        scaled_img = np.pad(input_img, padding, mode='constant', constant_values=0)
        return scaled_img.clip(min = 0)

    def resize_img_4d_01(self, input_img):
        c, h, w, d = input_img.shape
        scaled_img = snd.zoom(input_img, [c, self.input_size / h, self.input_size / w, self.depth_size / d], order=0)
        scaled_img = np.where(scaled_img > 0.5, 1, 0)
        return scaled_img

    def sample_conditions(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)
        input_files = [self.pair_files[index][0] for index in indexes]
        input_tensors = []
        for input_file in input_files:
            input_img = self.read_image(input_file, pass_scaler=self.full_channel_mask)
            input_img = np.expand_dims(input_img,0)
            # input_img = self.resize_img_4d(input_img)
            if self.transform is not None:
                input_img = self.transform(input_img).unsqueeze(0)
                input_tensors.append(input_img)
        return torch.cat(input_tensors, 0).cuda()

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, index):
        input_file, target_file = self.pair_files[index]
        input_img = self.read_image(input_file, pass_scaler=self.full_channel_mask)
        # input_img = self.resize_img_4d(input_img) # if not self.full_channel_mask else self.resize_img_4d(input_img)
        input_img = self.resize_img_4d_01(input_img)
        target_img = self.read_image(target_file)
        target_img = self.resize_img_4d(target_img)
        target_img = target_img / target_img.max()
        
        if self.transform is not None:
            input_img = self.transform(input_img)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)

        if self.combine_output:
            return torch.cat([target_img, input_img], 0)

        return {'input':input_img, 'target':target_img}


class NiftiPair3ImageGenerator(Dataset):
    def __init__(self,
                 tumor_folder: str,
                 ct_folder: str,
                 pet_folder: str,
                 input_size: int,
                 depth_size: int,
                 tumor_transform=None,
                 ct_transform=None,
                 pet_transform=None,
                 full_channel_mask=False,
                 combine_input=True
                 ):
        self.tumor_folder = tumor_folder
        self.ct_folder=ct_folder
        self.pet_folder = pet_folder
        self.pair_files = self.pair_file()
        self.input_size = input_size
        self.depth_size = depth_size
        self.scaler = MinMaxScaler()
        self.tumor_transform = tumor_transform
        self.ct_transform = ct_transform
        self.pet_transform = pet_transform
        self.full_channel_mask = full_channel_mask
        self.combine_input = combine_input

    def pair_file(self) -> list:
        tumor_files = sorted(glob(os.path.join(self.tumor_folder, '*')))
        pet_files = sorted(glob(os.path.join(self.pet_folder, '*_0000.nii.gz')))
        ct_files = sorted(glob(os.path.join(self.ct_folder, '*_0001.nii.gz')))
        pairs = []
        for tumor_file, ct_file ,pet_file in zip(tumor_files, ct_files, pet_files):
            assert int("".join(re.findall("\d", tumor_file))) == int("".join(re.findall("\d", ct_file))[:-4]) == int("".join(re.findall("\d", pet_file))[:-4])
            pairs.append((tumor_file, ct_file, pet_file))
        return pairs

    def read_image(self, file_path, pass_scaler=False):
        img = nib.load(file_path).get_fdata()
        img = img.clip(min=0)
        img = np.expand_dims(img, 0)
        return img

    def plot(self, index, n_slice=30):
        data = self[index]
        input_img = data['input']
        target_img = data['target']
        plt.subplot(1, 2, 1)
        plt.imshow(input_img[:, :, n_slice])
        plt.subplot(1, 2, 2)
        plt.imshow(target_img[:, :, n_slice])
        plt.show()

    def resize_img(self, img):
        h, w, d = img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(tensor=img[np.newaxis, ...])
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(cop(img))[0]
        return img

    def resize_img_4d(self, input_img):
        c, h, w, d = input_img.shape
        scaled_img = snd.zoom(input_img, [c, self.input_size / h, self.input_size / w, self.depth_size / d])
        return scaled_img.clip(min=0)

    def resize_img_4d_pad(self, input_img):
        c, h, w, d = input_img.shape
        pad_one_side = (self.input_size - h) // 2
        padding = [(0, 0), (pad_one_side, pad_one_side), (pad_one_side, pad_one_side), (pad_one_side, pad_one_side)]
        scaled_img = np.pad(input_img, padding, mode='constant', constant_values=0)
        return scaled_img.clip(min=0)

    def resize_img_4d_01(self, input_img):
        c, h, w, d = input_img.shape
        scaled_img = snd.zoom(input_img, [c, self.input_size / h, self.input_size / w, self.depth_size / d], order=0)
        scaled_img = np.where(scaled_img > 0.5, 1, 0)
        return scaled_img

    def sample_conditions(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)
        input_files = [self.pair_files[index][0] for index in indexes]
        input_tensors = []
        for input_file in input_files:
            input_img = self.read_image(input_file, pass_scaler=self.full_channel_mask)
            input_img = np.expand_dims(input_img, 0)
            # input_img = self.resize_img_4d(input_img)
            if self.transform is not None:
                input_img = self.transform(input_img).unsqueeze(0)
                input_tensors.append(input_img)
        return torch.cat(input_tensors, 0).cuda()

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, index):
        tumor_file, ct_file, pet_file = self.pair_files[index]
        tumor_img = self.read_image(tumor_file, pass_scaler=self.full_channel_mask)
        # input_img = self.resize_img_4d(input_img) # if not self.full_channel_mask else self.resize_img_4d(input_img)
        tumor_img = self.resize_img_4d_01(tumor_img)

        ct_img = self.read_image(ct_file)
        ct_img = self.resize_img_4d(ct_img)
        ct_img = ct_img / ct_img.max()


        pet_img = self.read_image(pet_file)
        pet_img = self.resize_img_4d(pet_img)
        pet_img = pet_img / pet_img.max()

        if self.tumor_transform is not None:
            tumor_img = self.tumor_transform(tumor_img)
        if self.ct_transform is not None:
            ct_img = self.ct_transform(ct_img)
        if self.pet_transform is not None:
            pet_img = self.pet_transform(pet_img)


        if self.combine_input:
            return {'input': torch.cat([tumor_img, ct_img], 0), 'target':pet_img}
        else:
            return {'tumor': tumor_img, 'ct':ct_img, 'pet': pet_img}



if __name__ =='__main__':
    # pet_transform = Compose([
    #     Lambda(lambda t: torch.tensor(t).float()),
    #     Lambda(lambda t: (t * 2) - 1),
    #     Lambda(lambda t: t.transpose(3, 1)),
    # ])
    # ct_transform = Compose([
    #     Lambda(lambda t: torch.tensor(t).float()),
    #     Lambda(lambda t: (t * 2) - 1),
    #     Lambda(lambda t: t.transpose(3, 1)),
    # ])
    # tumor_transform = Compose([
    #     Lambda(lambda t: torch.tensor(t).float()),
    #     Lambda(lambda t: t.transpose(3, 1)),
    # ])
    # ds=NiftiPair3ImageGenerator("/home/zyl/working/202406_01/Task107_hecktor2021/labelsTrain","/home/zyl/working/202406_01/Task107_hecktor2021/imagesTrain","/home/zyl/working/202406_01/Task107_hecktor2021/imagesTrain",128,128,tumor_transform,ct_transform,pet_transform)
    # print(len(ds))
    # print('input.shape:',ds[0]['input'].shape)
    # print('tumor.max/min:',ds[0]['input'][0].max(), ds[0]['input'][0].min())
    # print('ct.max/min:',ds[0]['input'][1].max(), ds[0]['input'][1].min())
    # print('target.shape:',ds[0]['target'].shape)
    # print('pet.max/min:',ds[0]['target'].max(), ds[0]['target'].min())
    tumorfolder = '/home/zyl/working/202406_01/Task107_hecktor2021/labelsTest/'
    ctfolder = '/home/zyl/working/202406_01/Task107_hecktor2021/imagesTest/'
    petfolder = '/home/zyl/working/202406_01/Task107_hecktor2021/imagesTest/'
    ds=SimpleDataset3D('/home/zyl/working/202406_01/scripts/runs/results/metrics/nocrop_ct',keyword='target')
    print(ds[0]['source'].shape)
    print(len(ds))