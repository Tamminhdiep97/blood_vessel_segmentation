import os
from os.path import join as opj
from glob import glob
import gc
import time
from datetime import datetime
import copy

from collections import defaultdict
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.cuda import amp
import torch.optim as optim
from tqdm import tqdm
from loguru import logger
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp

import config as conf
import utils


now = datetime.now()
dt_string = now.strftime('%d_%m_%Y_%H_%M')
name_weight_path = '{}_{}_{}'.format(dt_string, conf.model_name, conf.backbone_name)
weight_path = opj(conf.weight_path, name_weight_path) 
os.makedirs(weight_path, exist_ok=True)

utils.seed_everything(conf.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DiceLoss = smp.losses.DiceLoss(mode='binary')
BCELoss = smp.losses.SoftBCEWithLogitsLoss()


def criterion(y_pred, y_true):
    if conf.loss_func == "DiceLoss":
        return DiceLoss(y_pred, y_true)
    elif conf.loss_func == "BCELoss":
        y_true = y_true.unsqueeze(1)
        return BCELoss(y_pred, y_true)


def fetch_scheduler(optimizer):
    if conf.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=conf.T_max, 
                                                   eta_min=conf.min_lr)
    elif conf.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=conf.T_0, 
                                                             eta_min=conf.min_lr)
    elif conf.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=conf.min_lr,)
    elif conf.scheduler == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif conf.scheduler == None:
        return None
        
    return scheduler


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()
    
    dataset_size = 0
    running_loss = 0.0
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, (images, masks) in pbar:         
        images = images.to(device, dtype=torch.float)
        masks  = masks.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        with amp.autocast(enabled=True):
            y_pred = model(images)
            loss   = criterion(y_pred, masks)
            loss   = loss / conf.n_accumulate
            
        scaler.scale(loss).backward()
    
        if (step + 1) % conf.n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix( epoch=f'{epoch}',
                          train_loss=f'{epoch_loss:0.4f}',
                          lr=f'{current_lr:0.5f}',
                          gpu_mem=f'{mem:0.2f} GB')
    torch.cuda.empty_cache()
    gc.collect()
    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, optimizer, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    val_scores = []
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    for step, (images, masks) in pbar:        
        images  = images.to(device, dtype=torch.float)
        masks   = masks.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        y_pred  = model(images)
        loss    = criterion(y_pred, masks)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        y_pred = nn.Sigmoid()(y_pred)
        val_dice = utils.dice_coef(masks, y_pred).cpu().detach().numpy()
        val_jaccard = utils.iou_coef(masks, y_pred).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard])
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_memory=f'{mem:0.2f} GB')
    val_scores  = np.mean(val_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()
    return epoch_loss, val_scores


def run_training(model, optimizer, scheduler, device, num_epochs, train_loader, valid_loader):
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss      = np.inf
    best_epoch     = -1
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        logger.info('Epoch {}/{}'.format(epoch, num_epochs))
        train_loss = train_one_epoch(
                model, optimizer, scheduler,
                dataloader=train_loader, 
                device=device, epoch=epoch)
        
        val_loss, val_scores = valid_one_epoch(model, optimizer,
                                               valid_loader, device=device, 
                                               epoch=epoch)
        val_dice, val_jaccard = val_scores
        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['Valid Dice'].append(val_dice)
        history['Valid Jaccard'].append(val_jaccard)        
        logger.info('Valid Dice: {} | Valid Jaccard: {}'.format(round(val_dice, 4), round(val_jaccard, 4)))
        logger.info('Valid Loss: {}'.format(val_loss))
        
        # deep copy the model
        if val_loss <= best_loss:
            logger.info('Valid loss Improved ({} --> {})'.format(best_loss, val_loss))
            best_dice    = val_dice
            best_jaccard = val_jaccard
            best_loss = val_loss
            best_epoch   = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = "best_epoch.bin"
            torch.save(model.state_dict(), opj(weight_path, PATH))
            logger.info("Model Saved")
            
        last_model_wts = copy.deepcopy(model.state_dict())
        PATH = "last_epoch.bin"
        torch.save(model.state_dict(), opj(weight_path, PATH))
            
    end = time.time()
    time_elapsed = end - start
    logger.info('Training complete in {}h {}m {}s'.format(
        int(time_elapsed // 3600), int((time_elapsed % 3600) // 60),
        int((time_elapsed % 3600) % 60)
        )
    )
    logger.info("Best Loss: {}".format(best_loss))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


def main():
    train_groups = conf.train_groups
    valid_groups = conf.valid_groups
    gt_df = pd.read_csv(conf.gt_df)
    gt_df["img_path"] = gt_df["img_path"].apply(lambda x: os.path.join(conf.data_path, x))
    gt_df["msk_path"] = gt_df["msk_path"].apply(lambda x: os.path.join(conf.data_path, x))
    train_df = gt_df.query("group in @train_groups").reset_index(drop=True)
    valid_df = gt_df.query("group in @valid_groups").reset_index(drop=True)
    train_img_paths = train_df["img_path"].values.tolist()
    train_msk_paths = train_df["msk_path"].values.tolist()
    valid_img_paths = valid_df["img_path"].values.tolist()
    valid_msk_paths = valid_df["msk_path"].values.tolist()
    if conf.debug:
        train_img_paths = train_img_paths[:conf.train_bs*5]
        train_msk_paths = train_msk_paths[:conf.train_bs*5]
        valid_img_paths = valid_img_paths[:conf.valid_bs*3]
        valid_msk_paths = valid_msk_paths[:conf.valid_bs*3]

    model = utils.build_model(conf.backbone_name, conf.num_classes, device)

    data_transforms = utils.get_dataTransforms(conf.img_size)
    train_dataset = utils.BuildDataset(
            train_img_paths, train_msk_paths,
            transforms=data_transforms['train']
        )
    valid_dataset = utils.BuildDataset(
            valid_img_paths, valid_msk_paths,
            transforms=data_transforms['valid']
        )
    train_loader = DataLoader(
            train_dataset, batch_size=conf.valid_bs, num_workers=4, shuffle=False, pin_memory=True, drop_last=False
        )
    valid_loader = DataLoader(
            valid_dataset, batch_size=conf.valid_bs, num_workers=0, shuffle=False, pin_memory=True
        )

    optimizer = optim.Adam(model.parameters(), lr=conf.lr, weight_decay=conf.wd)
    scheduler = fetch_scheduler(optimizer)

    run_training(
            model, optimizer, scheduler, device, conf.epochs, train_loader, valid_loader
        )

if __name__ == '__main__':
    main()


