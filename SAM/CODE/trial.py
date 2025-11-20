import os
import numpy as np
from skimage import morphology, transform
import tifffile as tiff

source_path = "/mnt/40B2A1DBB2A1D5A6/lyx/TMP/neuron_tracing_from4090_1/Reference/DeepBranchTracer-new/DeepBranchTracer-main/data/Results/Drive/predicts_first/pre_centerline_test"
target_path = "/mnt/40B2A1DBB2A1D5A6/lyx/TMP/neuron_tracing_from4090_1/Reference/DeepBranchTracer-new/DeepBranchTracer-main/data/Results/Drive/predicts_first/reshape"
for f in os.listdir(source_path):
    if f.endswith('.tiff') or f.endswith('.tif'):
        path = os.path.join(source_path, f)
        img = tiff.imread(path)
        img = transform.resize(
            img,
            output_shape=(1, 584, 565),
            order=1,
            mode='edge',
            anti_aliasing=True,
            preserve_range=True
        )
        img = np.array(img, dtype=np.uint8)
        newpath = os.path.join(target_path, f)
        tiff.imwrite(newpath, img)
