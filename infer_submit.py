import os
from os.path import join as opj
import time
from glob import glob

# import cv2
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
import new_metric


utils.seed_everything(conf.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def check_size(bbox, canvas):
    x, y, w, h = bbox
    if x + w > canvas.shape[1]:
        bbox[2] = canvas.shape[1] - x
    if y + h > canvas.shape[0]:
        bbox[3] = canvas.shape[0] - y
    return bbox


def main(path_model):
    # model = utils.build_model(conf.backbone_name, conf.num_classes, device)
    # model.eval()
    valid_groups = conf.test_groups
    gt_df = pd.read_csv(conf.gt_df)
    # gt_df["img_path"] = gt_df["img_path"].apply(lambda x: os.path.join(conf.data_path, x))
    # gt_df["msk_path"] = gt_df["msk_path"].apply(lambda x: os.path.join(conf.data_path, x))
    valid_df = gt_df.query("group in @valid_groups").reset_index(drop=True)
    # valid_img_paths = valid_df["img_path"].values.tolist()
    # valid_msk_paths = valid_df["msk_path"].values.tolist()
    model = utils.load_model(
            conf.backbone_name, conf.num_classes, device, path_model
        )
    # list_image = glob(opj(conf.test_path, '*', 'images', '*.tif'))

    data_transformer = utils.get_dataTransforms_v2()
    # logger.info(valid_msk_paths)
    test_dataset = utils.SubmitTiledDataset(
            valid_df, tile_size=conf.test_img_size,
            transforms=data_transformer['valid'],
            overlap_pct=conf.overlap_pct
        )
    test_loader = DataLoader(
            test_dataset, batch_size=conf.test_bs,
            num_workers=4, shuffle=False, pin_memory=True
        )

    rles = []
    sigmoid_layer = nn.Sigmoid()
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Inference')
    # predict tiledataset
    thresh = 0.01
    time_ = 1
    while time_*thresh < conf.thresh:
        thresh_check = time_*10*thresh
        time_ += 1
        logger.info('Check score for thresh hold {}'.format(str(thresh_check)))
        count = 0
        position = []
        imgs = []
        img_fpaths = []
        bboxes = []
        shapes = []
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Inference')
        # for i, data in tqdm(enumerate(dataset)):
        for step, (data) in pbar:
            img, img_fpath, bbox, original_img_size, _ = data
            #print(img.shape)
            img = img.to(device, dtype=torch.float)
            with torch.no_grad():
                # rotate image
                # im_hf = torch.cat([torch.unsqueeze(torchvision.transforms.functional.hflip(im), dim=0) for im in img], dim=0)
                # im_vf = torch.cat([torch.unsqueeze(torchvision.transforms.functional.vflip(im), dim=0) for im in img], dim=0)
                # predict
                preds = (sigmoid_layer(model(img))>thresh_check).float()
                # preds_hf = model(im_hf)
                # preds_vf = model(im_vf)
                # deaumentation
                # preds_de_hf = torch.cat([torch.unsqueeze(torchvision.transforms.functional.hflip(im), dim=0) for im in preds_hf], dim=0)
                # preds_de_vf = torch.cat([torch.unsqueeze(torchvision.transforms.functional.hflip(im), dim=0) for im in preds_vf], dim=0)
                #pred_avg = (preds+preds_de_hf+preds_de_vf)/3
            img_fpaths.extend(img_fpath)
            for i in range(conf.test_bs):
                if i >= preds.shape[0]:
                    break
                imgs.append(utils.rle_encode((preds[i]).cpu().numpy().astype(np.uint8)))
                
                bboxes.append([int(bbox[0][i]), int(bbox[1][i]), int(bbox[2][i]), int(bbox[3][i])])
                shapes.append([int(original_img_size[0][i]), int(original_img_size[1][i])])
                position.append(conf.test_bs*step+i)
        
        unique_fpath = list(set(img_fpaths))
        position_index = dict()
        for fpath in unique_fpath:
            # position_index[fpath] = list(locate(img_fpaths, lambda x: x == fpath))
            position_index[fpath] = [idx for idx, value in enumerate(img_fpaths) if value == fpath]

        rles = []
        for key in position_index:
            shape = shapes[position_index[key][0]].copy()
            canvas = np.zeros(shape).astype(np.uint8)
            for index in position_index[key]:
                img =  utils.rle_decode(imgs[index], tuple(conf.test_img_size)).astype(np.uint8)
                # img = img.cpu().numpy().astype(np.uint8)
                bbox_ = bboxes[index]
                bbox = check_size(bbox_, canvas)
                x, y, w, h = bbox
                canvas[y:y+h, x:x+w] = np.logical_or(img[0:h, 0:w], canvas[y:y+h, x:x+w])
            # for index in position_index[key]
            rle = utils.rle_encode(utils.remove_small_objects(canvas, conf.min_size))
            rles.append(rle)

        ids = []
        for p_img in tqdm(position_index):
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
        # annotation_pd = pd.read_csv("gt_all.csv")  # , index_col=[0])
        # annotation_valid = annotation_pd.query("group in @valid_groups").reset_index(drop=True)
        # annotation_valid = annotation_valid.astype({'rle': 'str'})
        submission = pd.read_csv("submission.csv")  # , index_col=[0])
        submission = submission.astype({'rle': 'str'})
        if 'kidney_3_dense' in conf.test_groups:
            logger.info('*'*10)
            submission = submission.replace('kidney_3_sparse', 'kidney_3_dense', regex=True)
        list_index = valid_df['id'].tolist()
        submission = submission.set_index('id')
        submission = submission.loc[list_index]
        submission = submission.reset_index()
        # print(len(submission))
        t_start = time.time()
        logger.info('Starting compute score')
        # metric.score(
        #         valid_df,
        #         submission,
        #         row_id_column_name='id',
        #         rle_column_name='rle',
        #         tolerance=0.0,
        #         image_id_column_name='group',
        #         slice_id_column_name='slice',
        #         )
        logger.info(
                'Thresh: {}, Surface dice score: {}'.format(
                    str(thresh_check),
                    new_metric.compute_surface_dice_score(submission, valid_df)
                    )
                )
        t_end = time.time()
        logger.info('Time execute: {}s'.format(round(t_end-t_start)))


if __name__ == '__main__':
    path_model = opj(conf.weight_folder, '01_01_2024_14_53_Unet_resnext50_32x4d', 'last_epoch.bin')
    main(path_model)


