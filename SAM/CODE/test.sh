CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "/mnt/40B2A1DBB2A1D5A6/lyx/project/MedSam2/SAM-Unet/MODEL/model_saved/Mem-train-bk6_2025_03_27_15:13/Model/SAM2-UNet-view1-epoch-10.pth" \
--test_image_path "/mnt/40B2A1DBB2A1D5A6/lyx/project/MedSam2/SAM-Unet/DATA/test/jingbiaozhun/volumes_8bit" \
--output_path "/mnt/40B2A1DBB2A1D5A6/lyx/project/MedSam2/SAM-Unet/old/predicts"
 
