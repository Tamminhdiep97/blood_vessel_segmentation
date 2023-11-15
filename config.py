import os
from os.path import join as opj


seed = 42
name = 'unet-resnet50-512x512'
model_name = 'unet'
backbone_name = 'resnet50'
class_output = 2
data_folder = 'data'
train_folder = 'train'
test_folder = 'test'
valid_bs = 2
img_size  = [512, 512]

data_path = opj(os.getcwd(), data_folder)
train_path = opj(data_path, train_folder)
test_path = opj(data_path, test_folder)
