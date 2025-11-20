import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=10,help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',help='use cpu only')
parser.add_argument('--gpu_id', type=int, default=0, help='use gpu only')
parser.add_argument("--local_rank", type=int, default=-1)

# parser.add_argument("--resize_radio", type=float, default=1.0) # road
# parser.add_argument("--r_resize", type=float, default=10)

parser.add_argument("--resize_radio", type=float, default=1.0)  # drive 2 15 same with neuron #! big Neuron resize=1
parser.add_argument("--r_resize", type=float, default=15)

# parser.add_argument("--resize_radio", type=float, default=1.5) # chasedb 1.5 15
# parser.add_argument("--r_resize", type=float, default=15)
#TODO Model config
parser.add_argument(
    "--sam2_pretrain",
    default="/mnt/40B2A1DBB2A1D5A6/lyx/project/MODEL/Sam2/sam2_hiera_large.pt",
    help="path to sam2 weights"
)
parser.add_argument(
    "--adadim",
    default=32,
    help=" dims for adapter in sam2"
)
parser.add_argument(
    "--rfbdim",
    default=128,
    help="dim for rfb block"
)


# Datasets parameters DRIVE CHASEDB1 ROAD
parser.add_argument('--dataset_img_path', default = '/mnt/40B2A1DBB2A1D5A6/lyx/TMP/neuron_tracing_from4090_1/Reference/DeepBranchTracer-new/DeepBranchTracer-main/data/Neron2d_big/dataset/training_datasets/',help='Train datasets image root path')
parser.add_argument('--dataset_img_test_path', default = '/mnt/40B2A1DBB2A1D5A6/lyx/TMP/neuron_tracing_from4090_1/Reference/DeepBranchTracer-new/DeepBranchTracer-main/data/Neron2d_big/dataset/test_datasets/',help='Train datasets label root path')
parser.add_argument('--test_data_path', default = '/mnt/40B2A1DBB2A1D5A6/lyx/TMP/neuron_tracing_from4090_1/Reference/DeepBranchTracer-new/DeepBranchTracer-main/data/Neron2d_big/test/images/',help='Test datasets root path')
parser.add_argument('--test_data_mask_path', default = '/mnt/40B2A1DBB2A1D5A6/lyx/TMP/neuron_tracing_from4090_1/Reference/DeepBranchTracer-new/DeepBranchTracer-main/data/DRIVE/temp/mask/',help='Test datasets mask root path')


parser.add_argument('--predict_seed_path', default = '/mnt/40B2A1DBB2A1D5A6/lyx/TMP/neuron_tracing_from4090_1/Reference/DeepBranchTracer-new/DeepBranchTracer-main/data/Neron2d_big/test/',help='Seed root path')
parser.add_argument('--predict_centerline_path', default = '/mnt/40B2A1DBB2A1D5A6/lyx/TMP/neuron_tracing_from4090_1/Reference/DeepBranchTracer-new/DeepBranchTracer-main/data/Results/Neuron2d_big/Predicts/',help='Saved centerline result root path')
parser.add_argument('--predict_swc_path', default = '/mnt/40B2A1DBB2A1D5A6/lyx/TMP/neuron_tracing_from4090_1/Reference/DeepBranchTracer-new/DeepBranchTracer-main/data/Results/Neuron2d_big/pre_swc_test/',help='Saved swc result root path')

parser.add_argument('--batch_size', type=int, default=32, help='batch size of trainset')
parser.add_argument('--valid_rate', type=float, default=0.10, help='')
parser.add_argument('--data_shape', type=list, default=[64,64], help='')

parser.add_argument('--test_patch_height', default=64)
parser.add_argument('--test_patch_width', default=64)
parser.add_argument('--stride_height', default=48)
parser.add_argument('--stride_width', default=48)

# data in/out and dataset
parser.add_argument('--model_save_dir', default='/mnt/40B2A1DBB2A1D5A6/lyx/TMP/neuron_tracing_from4090_1/Reference/DeepBranchTracer-new/DeepBranchTracer-main/data/Results/Neuron2d_big/model/model2d_',help='save path of trained model')
parser.add_argument('--log_save_dir', default='/mnt/40B2A1DBB2A1D5A6/lyx/TMP/neuron_tracing_from4090_1/Reference/DeepBranchTracer-new/DeepBranchTracer-main/data/Results/Neuron2d_big/log/log2d_',help='save path of trained log')

# train
parser.add_argument('--epochs', type=int, default=10, metavar='N',help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',help='learning rate (default: 0.0001)')
# parser.add_argument('--early-stop', default=6, type=int, help='early stopping (default: 30)')
# parser.add_argument('--crop_size', type=int, default=48)
# parser.add_argument('--val_crop_max_size', type=int, default=96)
# parser.add_argument('--hidden_layer_size', type=int, default=1)
parser.add_argument('--vector_bins', type=int, default=50)
parser.add_argument('--train_seg', default=False, type=bool)

# test
# parser.add_argument('--use_amp', default=False, type=bool)
parser.add_argument('--train_or_test', default='inference_fastdeepbranchtracer')
parser.add_argument('--to_restore', default=True, type=bool)


args = parser.parse_args()