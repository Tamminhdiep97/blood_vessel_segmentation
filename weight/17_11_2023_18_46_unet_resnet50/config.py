import os
from os.path import join as opj


seed = 42
name = 'unet-resnet50-512x512'
model_name = 'unet'
backbone_name = 'resnet50'
class_output = 1
data_folder = 'data'
train_folder = 'train'
test_folder = 'test'
weight_folder = 'weight'
img_size  = [512, 512]

data_path = opj(os.getcwd(), data_folder)
train_path = opj(data_path, train_folder)
test_path = opj(data_path, test_folder)
weight_path = opj(os.getcwd(), weight_folder)


debug = False
train_bs = 12
valid_bs = 8
img_size = [512, 512]
epochs = 150
n_accumulate = max(1, 64//train_bs)
lr = 2e-3
scheduler = 'CosineAnnealingWarmRestarts'
min_lr = 1e-6
T_max = int(2279/(train_bs*n_accumulate)*epochs)+50
T_0 = 25
warmup_epochs = 5
wd = 1e-6
n_fold = 5
num_classes = 1

gt_df = opj(os.getcwd(), 'gt.csv')
train_groups = ["kidney_1_dense", "kidney_1_voi", "kidney_2", "kidney_3_sparse"]
valid_groups = ["kidney_3_dense"]
loss_func     = "DiceLoss"

