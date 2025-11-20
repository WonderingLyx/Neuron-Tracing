import torch
import numpy as np
import argparse
import cripser
from skimage import measure
from functools import reduce
import gudhi
import os
import cv2
import tifffile as tiff
import glob
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    '--gt',
    type=str,
    default='/mnt/40B2A1DBB2A1D5A6/lyx/TMP/neuron_tracing_from4090_1/Reference/DeepBranchTracer-new/DeepBranchTracer-main/data/single_neuron_3d/test/labels',
    help='Path to the directory containing ground-truth .tif files'
)
parser.add_argument(
    '--predicts',
    type=str,
    default='/mnt/40B2A1DBB2A1D5A6/lyx/TMP/neuron_tracing_from4090_1/Reference/DeepBranchTracer-new/DeepBranchTracer-main/data/Results/single_neuron_3d/predicts', 
    help='Path to the directory containing predicted .tiff files'
)
parser.add_argument(
    '--kernel_size', 
    type=int, 
    default=3, 
    help='Maxpooling kernel size'
)
parser.add_argument(
    '--device', 
    type=str, 
    default='cuda:0', 
    help='Device to run the evaluation (e.g., cuda:0 or cpu)'
)

args = parser.parse_args()

def iou(pre, label):
    eps = 1e-8
    if pre.sum() == 0:
        print('zero')
    return (pre * label).sum() / ((pre + label).sum() - (pre * label).sum() + eps)


def t_iou(pre, label):
    eps = 1e-8

    label_ori = label.clone()
    p1 = torch.nn.functional.max_pool3d(label, (3, 1, 1), 1, (1, 0, 0))
    p2 = torch.nn.functional.max_pool3d(label, (1, 3, 1), 1, (0, 1, 0))
    p3 = torch.nn.functional.max_pool3d(label, (1, 1, 3), 1, (0, 0, 1))
    label = torch.max(torch.max(p1, p2), p3) - label_ori

    return (pre * (label_ori + label)).sum() / ((pre * label + label_ori + pre - label_ori * pre).sum() + eps)

def junk_ratio(pre, label):
    difference = pre + label - 2 * pre * label
    difference = difference.view(pre.shape[0], -1)
    junks = label.view(label.shape[0], label.shape[1], -1).sum(axis=2)
    junks = (junks == 0).float()
    # axons = (junks != 0).float()
    # wrong_axons = (difference * axons).sum()
    wrong_junks = (difference * junks).sum()
    return wrong_junks / (difference.sum() + 1e-10)

def dice_error(input, target):
    smooth = 1.
    num = input.size(0)
    m1 = input.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
    # return (2. * intersection) / (m1.sum() + m2.sum())

def soft_skeletonize(x, thresh_width=5):
    '''
    Differenciable aproximation of morphological skelitonization operaton
    thresh_width - maximal expected width of vessel
    '''
    for i in range(thresh_width):
        p1 = torch.nn.functional.max_pool3d(x * -1, (3, 1, 1), 1, (1, 0, 0)) * -1
        p2 = torch.nn.functional.max_pool3d(x * -1, (1, 3, 1), 1, (0, 1, 0)) * -1
        p3 = torch.nn.functional.max_pool3d(x * -1, (1, 1, 3), 1, (0, 0, 1)) * -1
        min_pool_x = torch.min(torch.min(p1, p2), p3)
        contour = torch.nn.functional.relu(torch.nn.functional.max_pool3d(min_pool_x, (3, 3, 3), 1, 1) - min_pool_x)
        x = torch.nn.functional.relu(x - contour)
    return x

def positive_intersection(center_line, vessel):
    '''
    inputs shape  (batch, channel, height, width)
    intersection formalized by first ares
    x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
    '''
    clf = center_line.view(*center_line.shape[:2], -1)
    vf = vessel.view(*vessel.shape[:2], -1)

    intersection = (clf * vf).sum(-1)
    return (intersection.sum(0) + 1e-12) / (clf.sum(-1).sum(0) + 1e-12)
    

def soft_cldice_f1(pred, target):
    '''
    inputs shape  (batch, channel, height, width).
    calculate clDice acc
    '''
    target_skeleton = soft_skeletonize(target)
    cl_pred = soft_skeletonize(pred)
    # save = cl_pred.cpu().numpy()[0][0] * 255
    # tifffile.imsave('/media/root/data4/szy/validate/155829/whole_label/1.tiff', save.astype(np.uint8))
    # tifffile.imsave('/media/root/data4/szy/validate/155829/whole_label/1.tiff',
    #                 target.cpu().numpy()[0][0].astype(np.uint8))
    clrecall = positive_intersection(target_skeleton, pred)  # ClRecall
    recall = positive_intersection(target, pred)
    clacc = positive_intersection(cl_pred, target)
    acc = positive_intersection(pred, target)
    return clrecall[0], clacc[0], recall[0], acc[0]

def betti_csp(preds_cut, labels_cut, noise=1e-5):
    betti_pre = {0: 0, 1: 0, 2:0}
    betti_lb = {0: 0, 1: 0, 2:0}

    pred_res = cripser.computePH(preds_cut, maxdim=2)
    labels_res = cripser.computePH(labels_cut, maxdim=2)

    for diagram in pred_res:
        dim = diagram[0]  # 同调维度（0,1,2）
        birth = diagram[1]  # 出生时间
        death = diagram[2]  # 死亡时间
        if death - birth > noise:  # 过滤掉短暂存在的噪声特征
            betti_pre[dim] += 1
    
    for diagram in labels_res:
        dim = diagram[0]  # 同调维度（0,1,2）
        birth = diagram[1]  # 出生时间
        death = diagram[2]  # 死亡时间
        if death - birth > noise:  # 过滤掉短暂存在的噪声特征
            betti_lb[dim] += 1
    
    return betti_pre, betti_lb

def compute_betti_gud(matrix,min_pers=0,i=5):

    """
        Given a matrix representing a nii image compute the persistence diagram by using the Gudhi library (link)

        :param matrix: matrix encoding the nii image
        :type matrix: np.array

        :param min_pers: minimum persistence interval to be included in the persistence diagram
        :type min_pers: Integer

        :returns: Persistence diagram encoded as a list of tuples [d,x,y]=p where

            * d: indicates the dimension of the d-cycle p

            * x: indicates the birth of p

            * y: indicates the death of p
    """
    #save the dimenions of the matrix
    dims = matrix.shape
    size = reduce(lambda x, y: x * y, dims, 1)

    #create the cubica complex from the image
    cubical_complex = gudhi.CubicalComplex(dimensions=dims,top_dimensional_cells=np.reshape(matrix.T,size))
    #compute the persistence diagram
    if i == 5:
        pd = cubical_complex.persistence(homology_coeff_field=2, min_persistence=min_pers)
        return np.array(map(lambda row: [row[1][0],row[1][1]], pd))
    else:
        pd = cubical_complex.persistence(homology_coeff_field=2, min_persistence=min_pers)
        pd = cubical_complex.persistence_intervals_in_dimension(i)
        pd = np.array(list(map(lambda row: [row[0],row[1]], pd)))

        return len(pd)
    
def betti_gud(pred, label):
    betti_pre = {0: 0, 1: 0, 2:0}
    betti_lb = {0: 0, 1: 0, 2:0}

    for i in range(3):
        pre_x = compute_betti_gud(pred, i=i)
        betti_pre[i] = pre_x
    
    for i in range(3):
        lb_x = compute_betti_gud(label, i=i)
        betti_lb[i] = lb_x

    return betti_pre, betti_lb
   

def eval_two_volume_betti(preds, labels, noise=None, patch_size=(16,16,16), num=1):

    if noise == None:
        noise = 1e-5

    if preds.dtype is not np.float64:
        preds = preds.astype(np.float64) 
    
    if labels.dtype is not np.float64:
        labels = labels.astype(np.float64)

    beta = {
        0:0,
        1:0,
        2:0
    }

    for i in range(num):
        #preds_cut, labels_cut = ramdon_cut(preds, labels, patch_size=patch_size, indexs=patch_indx[:,i])
        
        #TODO crisper
        betti_pre, betti_lb = betti_csp(preds, labels)
        
        #TODO gudhi 
        #betti_pre, betti_lb = betti_gud(preds_cut, labels_cut)

        #print(betti_lb[1])
        for i in range(3):
            beta[i] += (abs(betti_pre[i] - betti_lb[i]))
    
    for i in range(3):
        beta[i] = beta[i] / num
        #print(f'beta{i}: {beta[i]}')

    return beta

def eval_two_volume_Euler(preds, labels, connectivity=4, patch_size=(16,16,16), num=1):

    if preds.dtype is not np.uint8:
        preds = preds.astype(np.uint8) 
    
    if labels.dtype is not np.uint8:
        labels = labels.astype(np.uint8)

    euler_diff = 0

    for i in range(num):
        #preds_cut, labels_cut = ramdon_cut(preds, labels, patch_size=patch_size, indexs=patch_indx[:,i])
        preds_euler = measure.euler_number(preds, connectivity=connectivity)
        lbs_euler = measure.euler_number(labels, connectivity=connectivity)

        euler_diff += abs(preds_euler  - lbs_euler)

        
    
    euler_diff = euler_diff / num

    return euler_diff

def eval_two_imgs_maxpool(gt, predict, pool_kernel, device):
    """
    计算单个标签和预测的评估指标
    """
    label = tiff.imread(gt)  # 读取标签
    pre = tiff.imread(predict)  # 读取预测

    k = pool_kernel
    kernel = (k, k, k)
    pre[pre < 125] = 0
    pre[pre >= 125] = 1
    label[label > 0] = 1
    pre = pre.astype(np.uint8)
    label = label.astype(np.uint8)

    pre = np.squeeze(pre)
    label = np.squeeze(label)

    pre = torch.Tensor(pre).view((1, 1, *pre.shape)).to(device)
    label = torch.Tensor(label).view((1, 1, *label.shape)).to(device)

    pre = torch.nn.functional.max_pool3d(pre, kernel, 1, 0)
    label = torch.nn.functional.max_pool3d(label, kernel, 1, 0)

    dice_score = dice_error(pre, label)
    total_loss_iou = iou(pre, label).cpu()
    total_loss_tiou = t_iou(pre, label).cpu()
    clrecall, clprecision, recall, precision = soft_cldice_f1(pre, label)
    cldice = (2. * clrecall * clprecision) / (clrecall + clprecision)

    pre = pre.cpu().numpy().squeeze()
    label = label.cpu().numpy().squeeze()

    bettis = eval_two_volume_betti(pre, label)

    eulers = eval_two_volume_Euler(pre, label)


    print('\nValidation IOU: {:.4f}\nT-IOU: {:.4f}\nClDice: {:.4f}\nClPrecision: {:.4f}\nClRecall: {:.4f}\nDice-score: {:.4f}\nPrecision: {:.4f}\nRecall: {:.4f}\nBetti0: {}\nBetti1: {}\nEuler: {}'
          .format(total_loss_iou, total_loss_tiou, cldice, clprecision, clrecall, dice_score, precision, recall, bettis[0], bettis[1], eulers))

    return {
        'iou': total_loss_iou,
        'tiou': total_loss_tiou,
        'cldice': cldice,
        'clprecision': clprecision,
        'clrecall': clrecall,
        'dice': dice_score,
        'precision': precision,
        'recall': recall,
        'betti0': bettis[0],
        'betti1': bettis[1],
        'euler': eulers
    }

if __name__ == "__main__":
    gt_list = os.listdir(args.gt)
    gt_list = [f for f in gt_list if f.endswith('.tiff') or f.endswith('.tif')]
    gt_list = sorted(gt_list)

    pred_list = os.listdir(args.predicts)
    pred_list = [f for f in pred_list if f.endswith('.tiff') or f.endswith('.tif')]
    pred_list = sorted(pred_list)

    assert len(gt_list) == len(pred_list) 

    device = torch.device(args.device)

    metrics = {
        "files": [], 
        'iou': [],
        'tiou': [],
        'cldice': [],
        'clprecision': [],
        'clrecall': [],
        'dice': [],
        'precision': [],
        'recall': [],
        'betti0': [],
        'betti1': [],
        'euler': []
    }
    for gt, pred in zip(gt_list, pred_list):
        file_name = gt.split('.')[0]
        metrics['files'].append(file_name)
        gt_path = os.path.join(args.gt, gt)
        pred_path = os.path.join(args.predicts, pred)
        eval_metrics = eval_two_imgs_maxpool(gt_path, pred_path, pool_kernel=args.kernel_size, device=device)
        for k in eval_metrics.keys():
            if isinstance(eval_metrics[k], torch.Tensor):
                eval_metrics[k] = eval_metrics[k].detach().cpu().numpy()
            metrics[k].append(eval_metrics[k])
    metrics['files'].append('Ave')
    for k in metrics.keys():
        if k == 'files':
            continue
        ave = np.array(metrics[k]).mean()
        metrics[k].append(ave)
    
    df = pd.DataFrame(metrics)
    df.to_excel(os.path.join(args.predicts, "Neuron3d_Eval_results.xlsx"), index=False)
        