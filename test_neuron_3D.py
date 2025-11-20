from configs import config_neuron_2d
from models.Models_3D import CSFL_Net_3D
import torch
import numpy as np
import os
import monai
from tools.Data_Loader_2d import Images_Dataset_folder_2d
from monai.inferers import sliding_window_inference
import torch.nn as nn
import argparse
import glob
import cv2
import tifffile as tiff

parser = argparse.ArgumentParser("test_neuron_3D")

parser.add_argument('--batch_size', type=int, default=8, help='batch size of trainset')
parser.add_argument('--epochs', type=int, default=30, metavar='N',help='number of epochs to train (default: 200)')
parser.add_argument('--n_threads', type=int, default=10,help='number of threads for data loading')
parser.add_argument('--vector_bins', type=int, default=50)
parser.add_argument('--data_shape', type=list, default=[64,64,64], help='')
parser.add_argument("--resize_radio", type=float, default=2.0)  # drive 2 15 same with neuron

parser.add_argument(
    "--pretrained_path",
    type=str,
    default="/mnt/40B2A1DBB2A1D5A6/lyx/TMP/neuron_tracing_from4090_1/Reference/DeepBranchTracer-new/DeepBranchTracer-main/data/Results/single_neuron_3d/model/model_0/15_8/epoch_15_batchsize_8.pth",
    help="path to pretrained model pth"
)
parser.add_argument(
    "--test_data_path",
    default="/mnt/40B2A1DBB2A1D5A6/lyx/TMP/neuron_tracing_from4090_1/Reference/DeepBranchTracer-new/DeepBranchTracer-main/data/single_neuron_3d/test/images",
    help="path to test imgs"
)
parser.add_argument(
    "--save_path",
    default="/mnt/40B2A1DBB2A1D5A6/lyx/TMP/neuron_tracing_from4090_1/Reference/DeepBranchTracer-new/DeepBranchTracer-main/data/Results/single_neuron_3d/predicts",
    help="path to save predicts"
)

args = parser.parse_args()

class Model(nn.Module):
    def __init__(self, in_channel, out_channel, freeze_net=False, weights_path=None, device=None, device_ids=[0]):
        super(Model, self).__init__()
        self.model = CSFL_Net_3D(in_channel, out_channel)
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        if device is not None:
            self.model = self.model.to(device)
        else:
            self.model = self.model.to("cuda:0")
        if weights_path is not None:
            assert os.path.exists(weights_path)
            checkpoint = torch.load(weights_path, weights_only=True)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
        else:
            raise ValueError(f"no Weights path")
        
    def forward(self, x):
        x = x.unsqueeze(1) 
        y_lab_pred, y_dis_pred = self.model(x, 'test_dis')
        
        
        return y_lab_pred



def test(args, device, device_ids=[0]):
    batch_size = args.batch_size
    epoch = args.epochs
    num_workers = args.n_threads
    vector_bins = args.vector_bins
    data_shape = args.data_shape
    resize_radio = args.resize_radio

    model = Model(1, 1, weights_path=args.pretrained_path, device=device, device_ids=device_ids)
    TEST_DIR = args.test_data_path
    test_image = glob.glob(TEST_DIR + '/*.tif')


    for test_img_dir in test_image:
        image_name = test_img_dir.split('/')[-1].split('.')[0]
        #TODO modify
        img_new = tiff.imread(test_img_dir).astype(np.float32)
        # img_new = cv2.imread(test_img_dir)
        # img_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
        origin_shape = img_new.shape
        # img_new = cv2.resize(
        #     img_new,
        #     (round(img_new.shape[0]*resize_radio), round(img_new.shape[1]*resize_radio)),
        #     interpolation=cv2.INTER_CUBIC
        # )
        #img_new = np.stack([img_new] * 3, axis=-1)
        #img_new = np.transpose(img_new, (2,0,1))
        img_new = img_new / 255
        if len(img_new.shape) != 5: #* NCHWD
            img_new = np.expand_dims(img_new, axis=0)
        
        with torch.no_grad():
            img_new = torch.from_numpy(img_new)
            img_new = img_new.unsqueeze(0)
            outputs = sliding_window_inference(
                inputs= img_new.to(torch.float32).to(device=device),
                roi_size= data_shape,
                sw_batch_size= 4,
                predictor= model,
                overlap= 0.5,
                mode="constant",
                padding_mode="constant",
                progress=True,
                device=device
            )
            

        outputs = outputs.detach().cpu().numpy()
        outputs = np.squeeze(outputs)
        # outputs = cv2.resize(
        #     outputs,
        #     origin_shape,
        #     interpolation=cv2.INTER_AREA
        # )
        outputs = (outputs * 255).astype(np.uint8)



        path = os.path.join(args.save_path, image_name + "_pro_lab.tif")
        tiff.imwrite(path, outputs)

            


        


if __name__ == "__main__":
    
    device_ids = [0,1]
    device = torch.device("cuda:0")
    test(args, device)
    