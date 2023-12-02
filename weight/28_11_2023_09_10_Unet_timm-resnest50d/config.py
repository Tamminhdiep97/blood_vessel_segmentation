import os
from os.path import join as opj


seed = 42
name = 'Unet-timm-resnext50d-512x512'
model_name = 'Unet'
backbone_name = 'timm-resnest50d'
class_output = 1
data_folder = 'blood-vessel-segmentation'
train_folder = 'train'
test_folder = 'test'
weight_folder = 'weight'
# img_size  = [512, 512]

data_path = opj(os.getcwd(), data_folder)
train_path = opj(data_path, train_folder)
test_path = opj(data_path, test_folder)
weight_path = opj(os.getcwd(), weight_folder)

train_method = 2
debug = False
train_bs = 8
valid_bs = 7
img_size = [512, 512]
epochs = 30
unfreezed_epoch = 10
n_accumulate = max(1, 64//train_bs)
lr = 1e-3
scheduler = 'CosineAnnealingLR'
min_lr = 5e-6
T_max = int((2279+1397+2217+1035)/(train_bs*n_accumulate)*epochs)+50
T_0 = 25
warmup_epochs = 5
wd = 1e-6
n_fold = 5
num_classes = 1

gt_df = opj(os.getcwd(), 'gt.csv')
train_groups = ["kidney_1_dense", "kidney_1_voi", "kidney_2", "kidney_3_sparse"]
valid_groups = ["kidney_3_dense"]
loss_func     = "MultiLoss_2"
thresh = 0.6
