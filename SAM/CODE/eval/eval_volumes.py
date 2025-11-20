from myutils import *
import pdb


def eval_two_volumes_maxpool(root, target, pool_kernel, device):
    pre = read_nifti(root)
    # pre = read_tiff_stack(root)
    label = read_nifti(target)
    k = pool_kernel
    s = max(1, k - 1)
    kernel = (k, k, k)
    stride = (s, s, s)
    # pre[pre < threshold] = 0
    # pre[pre >= threshold] = 1
    pre[pre > 0] = 1
    label[label > 0] = 1
    # 转Tensor时出现 TypeError: can't convert np.ndarray of type numpy.uint16.
    pre = pre.astype(np.uint8)
    label = label.astype(np.uint8)

    pre = torch.Tensor(pre).view((1, 1, *pre.shape)).to(device)
    # pre = torch_dilation(pre, 5)
    label = torch.Tensor(label).view((1, 1, *label.shape)).to(device)
    # label = torch_dilation(label, 5)

    pre = torch.nn.functional.max_pool3d(pre, kernel, 1, 0)
    label = torch.nn.functional.max_pool3d(label, kernel, 1, 0)
    # pre = torch.nn.functional.max_pool3d(pre, kernel, kernel, 0)
    # label = torch.nn.functional.max_pool3d(label, kernel, kernel, 0)

    dice_score = dice_error(pre, label)

    total_loss_iou = iou(pre, label).cpu()
    total_loss_tiou = t_iou(pre, label).cpu()
    clrecall, clacc, recall, acc = soft_cldice_f1(pre, label)
    cldice = (2. * clrecall * clacc) / (clrecall + clacc)

    # print('\n Validation IOU: {:.3f}\n T-IOU: {:.3f}'
    #       '\n ClDice: {:.3f} \n ClAcc: {:.3f} \n ClRecall: {:.3f} \n Dice-score: {:.3f}'
    #       .format(total_loss_iou, total_loss_tiou, cldice, acc, recall, dice_score))
#    print('\n Validation IOU: {}\n T-IOU: {}'
#          '\n ClDice: {} \n ClAcc: {} \n ClRecall: {} \n Dice-score: {}'
#          '\n Acc: {} \n Recall: {}'
#         .format(total_loss_iou, total_loss_tiou, cldice, clacc, clrecall, dice_score, acc, recall, ':.8f'))
#    print('\n{}\n{}\n{}\n{}\n{}\n{}'.format(cldice, clacc, clrecall, dice_score, acc, recall, ':.8f'))
    return {'iou': total_loss_iou,
            'tiou': total_loss_tiou,
            'cldice': cldice,
            'acc': clacc,
            'recall': clrecall,
            'score': dice_score}


def eval_two_volume_dirs_maxpool(target, root, data, pool_kernel, threshold, device):
    # pre = read_tiff_stack(join(root, data))
    # label = read_tiff_stack(join(target, data))
    pre = read_nifti(join(root, data))
    label = read_nifti(join(target, data))
    print(f"Prediction shape: {pre.shape}, Label shape: {label.shape}")
    if pre.shape != label.shape:
        raise ValueError(f"Prediction and label shapes do not match: {pre.shape} vs {label.shape}")
        
    k = pool_kernel
    s = max(1, k - 1)
    kernel = (k, k, k)
    stride = (s, s, s)
    # pre[pre < threshold] = 0
    # pre[pre >= threshold] = 1
    pre[pre > 0] = 1
    label[label > 0] = 1
    # 转Tensor时出现 TypeError: can't convert np.ndarray of type numpy.uint16.
    pre = pre.astype(np.uint8)
    label = label.astype(np.uint8)

    pre = torch.Tensor(pre).view((1, 1, *pre.shape)).to(device)
    # pre = torch_dilation(pre, 5)
    label = torch.Tensor(label).view((1, 1, *label.shape)).to(device)
    # label = torch_dilation(label, 5)

    pre = torch.nn.functional.max_pool3d(pre, kernel, 1, 0)
    label = torch.nn.functional.max_pool3d(label, kernel, 1, 0)

    dice_score = dice_error(pre, label)

    total_loss_iou = iou(pre, label).cpu()
    total_loss_tiou = t_iou(pre, label).cpu()
    recall, acc = soft_cldice_f1(pre, label)
    cldice = (2. * recall * acc) / (recall + acc)

def avg(num, total):
    return total / num


if __name__ == "__main__":

    # val_root = "./3D_predictions_test2Dloss_cldice/1_epoch_0_view_axial_label.tiff"
    # val_target = "./3D_predictions_test2Dloss_cldice/1_epoch_0_view_axial_prediction.tiff"
    # kernel_size = 2  # maxpooling kernel size
    # threshold = 125  # probability threshold of the positive class, no need for nnunet prediction
    # device = torch.device('cuda:1')
    # loss = eval_two_volumes_maxpool(val_target, val_root, kernel_size, device)

    val_dir = '/mnt/40B2A1DBB2A1D5A6/lyx/project/MedSam2/SAM-Unet/MODEL/model_trained/SAM2-Unet/Mem-num101_2025_03_18_15:20/Samples/epoch_29'
    device = torch.device('cuda', 0)
    labels = []
    predicts = []
    for f in os.listdir(val_dir):
        file_name = f.split('.')[0]
        if file_name.endswith('label'):
            label_path = os.path.join(val_dir, f)
            labels.append(label_path)
        if file_name.endswith('prediction'):
            predicts_path = os.path.join(val_dir, f)
            predicts.append(predicts_path)
    
    assert len(labels) == len(predicts)

    results = {'iou': [],
            'tiou': [],
            'cldice': [],
            'acc': [],
            'recall': [],
            'score': []}
    for pred, lb in zip(predicts, labels):
        res = eval_two_volumes_maxpool(pred, lb, pool_kernel=3, device=device)
        for key in results.keys():
            results[key].append(res[key].item())
    
    for k in results.keys():
        print(f'valuation {k} is {np.mean(results[k])}')