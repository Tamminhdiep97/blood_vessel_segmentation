import os
from os.path import join as opj


seed = 42
name = 'Unet-timm-resnest50d-256*256'
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
train_bs = 30
valid_bs = 68  # 20
img_size = [224, 224]
test_img_size = [256, 256]
test_bs = 32
epochs = 75
if debug:
    unfreezed_epoch = 2
else:
    unfreezed_epoch = 25
# n_accumulate = max(1, 1//train_bs)
n_accumulate = 1
lr = 1e-3
scheduler = 'CosineAnnealingLR'
min_lr = 1e-5
# T_max = int((kidney_1_dense+kidney_1_dense_xz+kidney_1_dense_zy+kidney_1_voi+kidney_3_dense_xz+kidney_3_dense_zy)/(train_bs*n_accumulate)*epochs)+50
T_0 = 30
T_max = 25
warmup_epochs = 5
wd = 1e-6
n_fold = 5
num_classes = 1

gt_df = opj(os.getcwd(), 'gt_all.csv')
train_groups = [
        "kidney_1_dense", "kidney_1_dense_xz", "kidney_1_dense_zy",
        "kidney_1_voi", "kidney_1_voi_xz", "kidney_1_voi_zy",
        ]
valid_groups = ["kidney_3_dense", 'kidney_3_dense_xz', 'kidney_3_dense_zy']
test_groups = ['kidney_3_dense']
loss_func     = "MultiLoss"
thresh = 0.95

overlap_pct = 25
min_size = 1

LOG_SYSTEM = 'SYSTEM'
LOG_LEVEL_SYSTEM = 100
TF_LOF_INTERVAL = 10

VERBOSE = False
LOG_LEVEL = 'INFO'
ROTATION = '500 MB'
RETENTION = '10 days'

if LOG_LEVEL == 'DEBUG':
    DEBUG = True
else:
    DEBUG = False

