import SimpleITK as sitk
import tifffile as tiff
import os
from tqdm import tqdm

# path = "/media/user/Expansion-1/20250728_16_57_23_YF2025072012_ZHAOHU_gulouyiyuan_nao_250605_2_Destripe_DONE/cut/size_1000_1000_500/volume-91203_Probabilities.tiff"
# new_path = "/media/user/Expansion-1/20250728_16_57_23_YF2025072012_ZHAOHU_gulouyiyuan_nao_250605_2_Destripe_DONE/cut/size_1000_1000_500/nii/volume-91203_Probabilities.nii.gz"

# reader = sitk.ImageFileReader()
# reader.SetFileName(path)
# tiff_image = reader.Execute()  # 读取为ITK图像对象
# labels = sitk.GetArrayFromImage(tiff_image)
# label_max = labels.max()
# labels[labels<label_max*0.5] = 0
# labels[labels>=label_max*0.5] = 1
# labels = sitk.GetImageFromArray(labels)
# sitk.WriteImage(labels, new_path)

def tiff2nii(input_dir, output_dir):
    assert os.path.exists(input_dir)
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.tiff') or f.endswith('.tif')]
    for f in tqdm(file_list):
        if "Probabilities" not in f:
            path = os.path.join(input_dir, f)
            name = f.split('.')[0]
            new_path = os.path.join(output_dir, name+'.nii.gz')
            
            reader = sitk.ImageFileReader()
            reader.SetFileName(path)
            tiff_image = reader.Execute()  # 读取为ITK图像对象
            sitk.WriteImage(tiff_image, new_path)
        
        else:
            path = os.path.join(input_dir, f)
            name = f.split('.')[0]
            new_path = os.path.join(output_dir, name+'.nii.gz')
            
            reader = sitk.ImageFileReader()
            reader.SetFileName(path)
            tiff_image = reader.Execute()  # 读取为ITK图像对象
            labels = sitk.GetArrayFromImage(tiff_image)
            label_max = labels.max()
            labels[labels<label_max*0.5] = 0
            labels[labels>=label_max*0.5] = 1
            labels = sitk.GetImageFromArray(labels)
            sitk.WriteImage(labels, new_path)

if __name__ == "__main__":
    input_dir = "/media/user/Expansion-1/20250729_13_14_54_YF2025072012_ZHAOHU_gulouyiyuan_nao_250604_5_Destripe_DONE/cut/volumes/trace/patchs/size_1000_1000_500"
    output_dir = "/media/user/Expansion-1/20250729_13_14_54_YF2025072012_ZHAOHU_gulouyiyuan_nao_250604_5_Destripe_DONE/cut/volumes/trace/patchs/size_1000_1000_500/nii"

    tiff2nii(input_dir, output_dir)