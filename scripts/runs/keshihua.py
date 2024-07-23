import vtk
import nibabel as nib

from vtkmodules.util import numpy_support
import numpy as np
img=nib.load('/home/zyl/working/202406_01/scripts/results/metrics/nocrop/sample_0.nii.gz')
# 将 nibabel 图像数据转换为 VTK 图像数据
numpy_data = img.get_fdata()
vtk_data = numpy_support.numpy_to_vtk(num_array=numpy_data.ravel(), deep=True, array_type=vtk.VTK_FLOAT)

# 创建 VTK 图像数据对象
vtk_image = vtk.vtkImageData()
vtk_image.SetSpacing(img.header.get_zooms()[:3])
vtk_image.SetDimensions(numpy_data.shape)
vtk_image.GetPointData().SetScalars(vtk_data)

# 创建映射器和体积
mapper = vtk.vtkSmartVolumeMapper()
mapper.SetInputData(vtk_image)

volume = vtk.vtkVolume()
volume.SetMapper(mapper)

# 创建渲染器、渲染窗口和交互器
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

renderer.AddVolume(volume)
renderer.SetBackground(1, 1, 1)  # 背景颜色为白色

renderWindow.Render()
renderWindowInteractor.Start()
