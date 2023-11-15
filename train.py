import os
from os.path import join as opj

import torch
import tqdm

import config as conf
import utils


utils.seed_everything(conf.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = utils.build_model(conf.backbone_name, conf.class_output, device)

model.train()
