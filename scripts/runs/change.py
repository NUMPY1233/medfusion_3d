#这个脚本用来验证肿瘤几何形状改变对结果的影响
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from pathlib import Path
import nibabel as nib

from medical_diffusion.data.datasets import NiftiPair3ImageGenerator
from medical_diffusion.models.pipelines import DiffusionPipeline


def visualize_3d_image(tumor,center):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(tumor[center[0], :, :], cmap='gray')
    plt.title('X Slice')
    plt.subplot(1, 3, 2)
    plt.imshow(tumor[:, center[1], :], cmap='gray')
    plt.title('Y Slice')
    plt.subplot(1, 3, 3)
    plt.imshow(tumor[:, :, center[2]], cmap='gray')
    plt.title('Z Slice')
    plt.show()
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




def create_ellipsoid(center, axes, shape, rotation_angles):
    """
    Generate a 3D ellipsoid in a 3D numpy array with rotation.

    :param center: A tuple (x, y, z) representing the center of the ellipsoid.
    :param axes: A tuple (a, b, c) representing the radii along each axis.
    :param shape: A tuple (dim_x, dim_y, dim_z) representing the dimensions of the array.
    :param rotation_angles: A tuple (theta_x, theta_y, theta_z) representing the rotation angles (in degrees) around each axis.
    :return: A 3D numpy array with the rotated ellipsoid.
    """
    grid = np.zeros(shape, dtype=float)
    x_range = np.arange(shape[0])
    y_range = np.arange(shape[1])
    z_range = np.arange(shape[2])

    x, y, z = np.meshgrid(x_range, y_range, z_range, indexing='ij')

    # Convert angles to radians
    theta_x, theta_y, theta_z = np.deg2rad(rotation_angles)

    # Rotation matrices around each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])

    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])

    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])

    # Complete rotation matrix
    R = np.dot(Rz, np.dot(Ry, Rx))

    # Apply rotation and translate back to center
    xyz = np.vstack([x.ravel() - center[0], y.ravel() - center[1], z.ravel() - center[2]])
    xyz_rotated = np.dot(R, xyz)

    # Ellipsoid equation in the rotated frame
    inside = ((xyz_rotated[0, :] / axes[0]) ** 2 +
              (xyz_rotated[1, :] / axes[1]) ** 2 +
              (xyz_rotated[2, :] / axes[2]) ** 2 <= 1)

    grid.ravel()[inside] = 1  # Set points inside the ellipsoid to 1

    return grid


# Image dimensions
dim_x, dim_y, dim_z = 128, 128, 128

# Ellipsoid properties
center = (65, 58, 64)  # Arbitrary center position
axes = (5,10, 15)  # Semi-axes of the ellipsoid (a, b, c)
rotation_angles = (0, 0, 0)
# Create the ellipsoid at the specified position
tumor = create_ellipsoid(center, axes, (dim_x, dim_y, dim_z), rotation_angles)
visualize_3d_image(tumor,center)
tumor= resize_img_4d_01(tumor)
print(tumor.shape)
print(tumor.max(), tumor.min())
# Display the middle slices to verify



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
ct=ds[70]['ct']
tumor=tumor
condition = torch.cat((tumor, ct), dim=0)   #with_ct
# condition=tumor #without_ct
print(condition.shape)
condition.unsqueeze_(0)

device = torch.device('cuda:1')
condition=condition.to(device)
# pipeline = DiffusionPipeline.load_from_checkpoint(
#     "/home/zyl/working/202406_01/scripts/runs/LDM_VQGAN2/2024_06_19_130913/epoch=1529-step=153000.ckpt")
pipeline = DiffusionPipeline.load_from_checkpoint(
    "/home/zyl/working/202406_01/scripts/runs/LDM_VQGAN2/2024_06_19_130913/epoch=1529-step=153000.ckpt")
pipeline.to(device)

results = pipeline.sample(1, (4, 16, 16, 16), condition=condition, guidance_scale=1, steps=250,
                              use_ddim=True)
results = (results + 1) / 2  # Transform from [-1, 1] to [0, 1]
results = results.clamp(0, 1)
path_out = Path(path_out)
path_out.mkdir(parents=True, exist_ok=True)

sample_img1 = results.squeeze(0).squeeze(0).detach().cpu().numpy()
nifti_img_s = nib.Nifti1Image(sample_img1, affine=np.eye(4))
nib.save(nifti_img_s, path_out / f'sample9.nii.gz')
tumor_img1= tumor.squeeze(0).detach().cpu().numpy()
nifti_img_tumor = nib.Nifti1Image(tumor_img1, affine=np.eye(4))
nib.save(nifti_img_tumor, path_out / f'tumor9.nii.gz')




