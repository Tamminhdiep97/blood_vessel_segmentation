import os
from os.path import join as opj
from glob import glob

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from loguru import logger
import numpy as np
import pandas as pd

import config as conf
import utils


utils.seed_everything(conf.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    model = utils.build_model(conf.backbone_name, conf.class_output, device)
    model.eval()
    
    list_image = glob(opj(conf.test_path, '*', 'images', '*.tif'))

    data_transformer = utils.get_dataTransforms(conf.img_size)
    test_dataset = utils.BuildDataset(list_image, [], transforms=data_transformer['valid'])
    test_loader = DataLoader(test_dataset, batch_size=conf.valid_bs, num_workers=4, shuffle=False, pin_memory=True)

    rles = []
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Inference')
    for step, (images, shapes) in pbar:
        shapes = shapes.numpy()
        images = images.to(device, dtype=torch.float)
        with torch.no_grad():
            preds = model(images)
            preds = (nn.Sigmoid()(preds)>0.5).double()
        preds = preds.cpu().numpy().astype(np.uint8)

        for pred, shape in zip(preds, shapes):
            pred = cv2.resize(pred[0], (shape[1], shape[0]), cv2.INTER_NEAREST)
            rle = utils.rle_encode(pred)
            rles.append(rle)

    ids = []
    for p_img in tqdm(list_image):
        path_ = p_img.split(os.path.sep)
        # parse the submission ID
        dataset = path_[-3]
        slice_id, _ = os.path.splitext(path_[-1])
        ids.append(f"{dataset}_{slice_id}")

    submission = pd.DataFrame.from_dict(
            {
                'id': ids,
                'rle': rles
            }
        )
    submission.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    main()


