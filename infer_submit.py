import os
from os.path import join as opj
import time
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
import metric

utils.seed_everything(conf.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def main(path_model):
    # model = utils.build_model(conf.backbone_name, conf.num_classes, device)
    # model.eval()
    valid_groups = conf.valid_groups
    gt_df = pd.read_csv(conf.gt_df)
    gt_df["img_path"] = gt_df["img_path"].apply(lambda x: os.path.join(conf.data_path, x))
    gt_df["msk_path"] = gt_df["msk_path"].apply(lambda x: os.path.join(conf.data_path, x))
    valid_df = gt_df.query("group in @valid_groups").reset_index(drop=True)
    valid_img_paths = valid_df["img_path"].values.tolist()
    valid_msk_paths = valid_df["msk_path"].values.tolist()
    model = utils.load_model(
            conf.backbone_name, conf.num_classes, device, path_model
        )
    # list_image = glob(opj(conf.test_path, '*', 'images', '*.tif'))

    data_transformer = utils.get_dataTransforms(conf.img_size)
    # logger.info(valid_msk_paths)
    test_dataset = utils.BuildDataset(
            valid_img_paths, [], transforms=data_transformer['valid']
        )
    test_loader = DataLoader(
            test_dataset, batch_size=conf.valid_bs,
            num_workers=4, shuffle=False, pin_memory=True
        )

    rles = []
    sigmoid_layer = nn.Sigmoid()
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Inference')
    for step, (images, shapes) in pbar:
        # print(shapes)
        shapes = shapes.numpy()
        images = images.to(device, dtype=torch.float)
        with torch.no_grad():
            preds = model(images)
            preds = (sigmoid_layer(preds)>conf.thresh).double()
        preds = preds.cpu().numpy().astype(np.uint8)

        for pred, shape in zip(preds, shapes):
            # print(pred[0])
            # print(shape[0], shape[1])
            pred = cv2.resize(pred[0], (shape[1], shape[0]), cv2.INTER_NEAREST)
            rle = utils.rle_encode(pred)
            rles.append(rle)

    ids = []
    for p_img in tqdm(valid_msk_paths):
        path_ = p_img.split(os.path.sep)
        # parse the submission ID
        dataset = path_[-3]
        slice_id, _ = os.path.splitext(path_[-1])
        ids.append(f"{dataset}_{slice_id}")

    submission = pd.DataFrame.from_dict(
            {
                'id': ids,
                'rle': rles,
            }
        )
    submission.to_csv("submission.csv", index=False)
    annotation_pd = pd.read_csv("gt.csv")  # , index_col=[0])
    annotation_valid = annotation_pd.query("group in @valid_groups").reset_index(drop=True)
    annotation_valid = annotation_valid.astype({'rle': 'str'})
    submission = pd.read_csv("submission.csv")  # , index_col=[0])
    # submission = submission.astype({'rle': 'str'})
    # print(len(submission))
    # print(submission.head())
    # print(annotation_valid.head())
    t_start = time.time()
    metric.score(
            valid_df,
            submission,
            row_id_column_name='id',
            rle_column_name='rle',
            tolerance=0.0,
            image_id_column_name='group',
            slice_id_column_name='slice',
            )
    t_end = time.time()
    logger.info('Time execute: {}s'.format(round(t_end-t_start)))


if __name__ == '__main__':
    path_model = opj(conf.weight_folder, '17_11_2023_00_20_unet_resnet50', 'best_epoch.bin')
    main(path_model)


