import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import tifffile as tiff
from tqdm import tqdm

imgs_labels_path = "/mnt/40B2A1DBB2A1D5A6/lyx/TMP/neuron_tracing_from4090_1/data/OpenSource/A neuronal imaging dataset for deep learning in the reconstruction of single-neuron axons/images/images-8bit"
swc_dir = "/mnt/40B2A1DBB2A1D5A6/lyx/TMP/neuron_tracing_from4090_1/data/OpenSource/A neuronal imaging dataset for deep learning in the reconstruction of single-neuron axons/swc/swc"

#! if split temperal
labels_path = '/mnt/40B2A1DBB2A1D5A6/lyx/TMP/neuron_tracing_from4090_1/data/OpenSource/A neuronal imaging dataset for deep learning in the reconstruction of single-neuron axons/mask/mask'

target_dir = "/mnt/40B2A1DBB2A1D5A6/lyx/TMP/neuron_tracing_from4090_1/Reference/DeepBranchTracer-new/DeepBranchTracer-main/data/single_neuron_3d"
TEST_RATIO = 0.2

if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(os.path.join(target_dir, 'training/images'))
    os.makedirs(os.path.join(target_dir, 'training/labels'))
    os.makedirs(os.path.join(target_dir, 'training/swc'))
    
    os.makedirs(os.path.join(target_dir, 'test/images'))
    os.makedirs(os.path.join(target_dir, 'test/labels'))
    os.makedirs(os.path.join(target_dir, 'test/swc'))

    os.makedirs(os.path.join(target_dir, 'dataset'))

imgs = []
labels = []
swc = os.listdir(swc_dir)

for f in os.listdir(imgs_labels_path):
    if f.endswith('.tiff') or f.endswith('.tif'):
        if 'Probabilities' in f:
            labels.append(f)
        else:
            imgs.append(f)
#! if split
labels = os.listdir(labels_path)

# imgs = sorted(imgs, key=lambda x: int(x.split('.')[0]))
# labels = sorted(labels, key=lambda x: int(x.split('_')[0]))
# swc = sorted(swc, key=lambda x: int(x.split('_')[0]))
imgs = sorted(imgs)
labels = sorted(labels)
swc = sorted(swc)

assert len(imgs) == len(labels)
assert len(labels) == len(swc)

imgs_tr, imgs_te, labels_tr, labels_te, swc_tr, swc_te = train_test_split(
    np.array(imgs),
    np.array(labels),
    np.array(swc),
    test_size=TEST_RATIO,
    random_state=42,
    shuffle=True
)

for f in tqdm(imgs_tr, desc='train_dataset'):
    path = os.path.join(imgs_labels_path, f)
    file_name, appendix = f.split('.')
    new_path = os.path.join(target_dir+'/training/images', f)
    shutil.copy(path, new_path)

    # label_name = file_name + '_Probabilities.tiff'
    # label_path = os.path.join(imgs_labels_path, label_name)
    # new_path = os.path.join(target_dir+'/training/labels', label_name)

    label_name = file_name + '.tif'
    label_path = os.path.join(labels_path, label_name)
    new_path = os.path.join(target_dir+'/training/labels', label_name)

    lb = tiff.imread(label_path)
    label_max = lb.max()
    assert label_max < 256
    lb[lb<label_max*0.5] = 0
    lb[lb>=label_max*0.5] = 255
    lb = lb.astype(np.uint8)
    
    tiff.imwrite(new_path, lb)

    # swc_name = file_name + '_Probabilities.swc'
    # swc_path = os.path.join(swc_dir, swc_name)
    # new_path = os.path.join(target_dir+'/training/swc', swc_name)
    # shutil.copy(swc_path, new_path)

    swc_name = file_name + '.swc'
    swc_path = os.path.join(swc_dir, swc_name)
    new_path = os.path.join(target_dir+'/training/swc', swc_name)
    shutil.copy(swc_path, new_path)

for f in tqdm(imgs_te, desc='test_dataset'):
    path = os.path.join(imgs_labels_path, f)
    file_name, appendix = f.split('.')
    new_path = os.path.join(target_dir+'/test/images', f)
    shutil.copy(path, new_path)

    # label_name = file_name + '_Probabilities.tiff'
    # label_path = os.path.join(imgs_labels_path, label_name)
    # new_path = os.path.join(target_dir+'/test/labels', label_name)

    label_name = file_name + '.tif'
    label_path = os.path.join(labels_path, label_name)
    new_path = os.path.join(target_dir+'/test/labels', label_name)

    lb = tiff.imread(label_path)
    label_max = lb.max()
    assert label_max < 256
    lb[lb<label_max*0.5] = 0
    lb[lb>=label_max*0.5] = 255
    lb = lb.astype(np.uint8)
    
    tiff.imwrite(new_path, lb)

    # swc_name = file_name + '_Probabilities.swc'
    # swc_path = os.path.join(swc_dir, swc_name)
    # new_path = os.path.join(target_dir+'/test/swc', swc_name)
    # shutil.copy(swc_path, new_path)

    swc_name = file_name + '.swc'
    swc_path = os.path.join(swc_dir, swc_name)
    new_path = os.path.join(target_dir+'/test/swc', swc_name)
    shutil.copy(swc_path, new_path)

# for f in tqdm(labels_tr, desc='labels_tr'):
#     path = os.path.join(imgs_labels_path, f)
#     new_path = os.path.join(target_dir+'/training/labels', f)

#     lb = tiff.imread(path)
#     label_max = lb.max()
#     lb[lb<label_max*0.5] = 0
#     lb[lb>=label_max*0.5] = 255
#     lb = lb.astype(np.uint8)
    
#     tiff.imwrite(new_path, lb)

# for f in tqdm(swc_tr, desc='swc_tr'):
#     path = os.path.join(swc_path, f)
#     new_path = os.path.join(target_dir+'/training/swc', f)
#     shutil.copy(path, new_path)

# for f in tqdm(imgs_te, desc="imgs_te"):
#     path = os.path.join(imgs_labels_path, f)
#     new_path = os.path.join(target_dir+'/test/images', f)
#     shutil.copy(path, new_path)

# for f in tqdm(labels_te, desc="labels_te"):
#     path = os.path.join(imgs_labels_path, f)
#     new_path = os.path.join(target_dir+'/test/labels', f)

#     lb = tiff.imread(path)
#     label_max = lb.max()
#     lb[lb<label_max*0.5] = 0
#     lb[lb>=label_max*0.5] = 255
#     lb = lb.astype(np.uint8)
    
#     tiff.imwrite(new_path, lb)

# for f in tqdm(swc_te, desc='swc_te'):
#     path = os.path.join(swc_path, f)
#     new_path = os.path.join(target_dir+'/test/swc', f)
#     shutil.copy(path, new_path)






