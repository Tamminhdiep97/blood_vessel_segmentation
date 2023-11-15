import torch
import segmentation_models_pytorch as smp
import cv2
from loguru import logger
import numpy as np


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = np.tile(img[...,None], [1, 1, 3]) # gray to rgb
    img = img.astype('float32') # original is uint16
    mx = np.max(img)
    if mx:
        img /= mx # scale image to [0, 1]
    return img


def load_msk(path):
    msk = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    msk = msk.astype('float32')
    msk /= 255.0
    return msk


def build_model(backbone, num_classes, device):
    model = smp.Unet(
        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_name=backbone,
        # use `imagenet` pre-trained weights for encoder initialization
        encoder_weights=None,
        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        in_channels=3,
        # model output channels (number of classes in your dataset)
        classes=num_classes,
        activation=None,
    )
    model.to(device)
    return model


def load_model(backbone, num_classes, device, path):
    model = build_model(backbone, num_classes, device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, msk_paths=[], transforms=None):
        self.img_paths  = img_paths
        self.msk_paths  = msk_paths
        self.transforms = transforms
 
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path  = self.img_paths[index]
        img = load_img(img_path)

        if len(self.msk_paths)>0:
            msk_path = self.msk_paths[index]
            msk = load_msk(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img  = data['image']
                msk  = data['mask']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)
        else:
            orig_size = img.shape
            if self.transforms:
                data = self.transforms(image=img)
                img  = data['image']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img), torch.tensor(np.array([orig_size[0], orig_size[1]]))
