import os
from os.path import join as opj

import rasterio
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import segmentation_models_pytorch as smp
import cv2
from loguru import logger
import numpy as np
import pandas as pd
from functools import lru_cache


@lru_cache(maxsize=64)
def ropen(img_fpath):
    return rasterio.open(img_fpath)



def remove_small_objects(img, min_size):
    # Find all connected components (labels)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

    # Create a mask where small objects are removed
    new_img = np.zeros_like(img)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_size:
            new_img[labels == label] = 1
    return new_img


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
    torch.backends.cudnn.benchmark = False


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


def get_dataTransforms(img_size):
    data_transforms = {
        "train": A.Compose(
            [
                A.Resize(
                    *img_size, interpolation=cv2.INTER_NEAREST
                ),
                A.HorizontalFlip(p=0.8),
                A.VerticalFlip(p=0.8),
                A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0.5, shift_limit=0.5, p=0.8),

                # A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
                # A.RandomCrop(height=320, width=320, always_apply=True),
                # A.IAAAdditiveGaussianNoise(p=0.2),
                A.Perspective(p=0.8),

                A.OneOf(
                    [
                        A.RandomBrightnessContrast(p=1),
                        A.RandomGamma(p=1),
                    ],
                    p=0.8,
                ),

                A.OneOf(
                    [
                        A.Sharpen(p=1),
                        A.Blur(blur_limit=3, p=1),
                        A.MotionBlur(blur_limit=3, p=1),
                    ],
                    p=0.8,
                ),

                A.OneOf(
                    [
                        A.RandomContrast(p=1),
                        A.HueSaturationValue(p=1),
                    ],
                    p=0.8,
                ),
            ],
            p=1.0
        ),
        "valid": A.Compose(
            [
                A.Resize(
                    *img_size, interpolation=cv2.INTER_NEAREST
                ),
            ],
            p=1.0
        )
        # "valid": False
    }
    return data_transforms


def get_dataTransforms_v2():
    data_transforms = {
        "train": A.Compose(
            [
                # A.ToRGB(),
                # A.Resize(
                #     *img_size, interpolation=cv2.INTER_NEAREST
                # ),
                A.VerticalFlip(p=0.6),
                A.HorizontalFlip(p=0.6),
                A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0.5, shift_limit=0.5, p=0.6, border_mode=0),

                # A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
                # A.RandomCrop(height=320, width=320, always_apply=True),
                # A.IAAAdditiveGaussianNoise(p=0.2),
                A.Perspective(p=0.6),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(p=1),
                        A.RandomGamma(p=1),
                    ],
                    p=0.6,
                ),

                A.OneOf(
                    [
                        A.Sharpen(p=1),
                        A.Blur(blur_limit=3, p=1),
                        A.MotionBlur(blur_limit=3, p=1),
                    ],
                    p=0.6,
                ),
                ToTensorV2()
            ],
            p=1.0
        ),
        "valid": A.Compose(
            [
                # A.ToRGB(),
                ToTensorV2()
                # A.Resize(
                #     *img_size, interpolation=cv2.INTER_NEAREST
                # ),
            ],
            p=1.0
        )
        # "valid": False
    }
    return data_transforms


def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.unsqueeze(1).to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice


def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.unsqueeze(1).to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou


def build_model(backbone, num_classes, device):
    # model = smp.Unet(
    #     # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #     encoder_name=backbone,
    #     # use `imagenet` pre-trained weights for encoder initialization
    #     encoder_weights='imagenet',
    #     # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #     in_channels=3,
    #     # model output channels (number of classes in your dataset)
    #     classes=num_classes,
    #     activation=None,
    # )
    model = smp.Unet(
        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_name=backbone,
        # use `imagenet` pre-trained weights for encoder initialization
        encoder_weights='imagenet',
        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        in_channels=1,
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


# def rle_encode(img):
#     '''
#     img: numpy array, 1 - mask, 0 - background
#     Returns run length as string formated
#     '''
#     pixels = img.flatten()
#     pixels = np.concatenate([[0], pixels, [0]])
#     runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
#     runs[1::2] -= runs[::2]
#     rle = ' '.join(str(x) for x in runs)
#     if rle == '':
#         rle = '1 0'
#     return rle


def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = ' '.join(str(x) for x in runs)
    if rle == '':
        rle = '1 0'
    return rle


def rle_decode(mask_rle: str, img_shape: tuple = None) -> np.ndarray:
    seq = mask_rle.split()
    starts = np.array(list(map(int, seq[0::2])))
    lengths = np.array(list(map(int, seq[1::2])))
    assert len(starts) == len(lengths)
    ends = starts + lengths
    img = np.zeros((np.product(img_shape),), dtype=np.uint8)
    for begin, end in zip(starts, ends):
        img[begin:end] = 1
    # https://stackoverflow.com/a/46574906/4521646
    img.shape = img_shape
    return img


def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        A.RandomCrop(height=320, width=320, always_apply=True),
        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(train_transform)


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


class SenNetHOATiledDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            df_data: pd.DataFrame,
            path_img_dir: str,
            transforms=None,
            tile_size=None,
            overlap_pct: float = 0.2,
            empty_tile_pct: float = 0.0,
            cache_dir: str = None
    ):
        """
        Generates an in-memory tiled dataset using a sliding window approach (common in geospatial processing communities)

        Args:
            df_data (pd.DataFrame): Training dataframe.
            path_img_dir (str): Path to root directory containing data.
            transforms (torchvision.transforms.Transform): Albumentation transforms.
            tile_size (Optional[List[int]]): List describing the tile size. If None, tiling is not performed and the
                full scene is utilized.
            overlap_pct (float): Percentage of tile size to use as overlap between tiles. This effectively
                controls the "stride" of the window as it's slid across the full scene.
            empty_tile_pct (float): Percentage of final dataset that should be empty tiles. Default is 0.0 (no empty tiles)
        """
        self.path_img_dir = path_img_dir
        self.transforms = transforms
        self.data = df_data
        self.samples = []
        self.tile_size = np.array(tile_size) if tile_size is not None else None
        self.overlap_pct = overlap_pct
        self.empty_tile_pct = empty_tile_pct
        self.cache_dir = cache_dir
        self.value_dict = dict()

        for _, row in self.data.iterrows():
            # p_img = os.path.join(self.path_img_dir, row["group"], "images", f'{row["slice"]}.tif')
            p_img = row['img_path']

            with rasterio.open(p_img) as reader:
                width, height = reader.width, reader.height
                img = reader.read()
                px_max, px_min = img.max(), img.min()
                self.check_value(row['group'], px_max, px_min)

            if not os.path.isfile(p_img):
                continue
            self.samples.append((p_img, row['group'], row['rle'], [], [], 1, [width, height]))

        if self.tile_size is not None:
            empty = 0
            nonempty = 0

            empty_tiles = []
            populated_tiles = []

            for file_path, group, rle, _, _, _, img_dims in tqdm(self.samples, total=len(self.samples), desc='Generating tiles'):
                width, height = img_dims

                mask = rle_decode(rle, img_shape=[height, width])

                min_overlap = float(overlap_pct) * 0.01
                max_stride = self.tile_size * (1.0 - min_overlap)
                num_patches = np.ceil(np.array([height, width]) / max_stride).astype(np.int64)
                # compute the cutoff points for the x-y dimensions
                starts = [np.int64(np.linspace(0, width - self.tile_size[1], num_patches[1])),
                          np.int64(np.linspace(0, height - self.tile_size[0], num_patches[0]))]
                stops = [starts[0] + self.tile_size[0], starts[1] + self.tile_size[1]]
                for y1, y2 in zip(starts[1], stops[1]):
                    for x1, x2 in zip(starts[0], stops[0]):
                        this_region = mask[y1:y2, x1:x2]
                        is_empty = np.all(this_region == 0)

                        if self.empty_tile_pct == 0.0:
                            is_empty = is_empty or (this_region.sum() < (0.05 * self.tile_size[0]))

                        if is_empty:
                            empty += 1
                            empty_tiles.append((file_path, group, rle, [x1, y1, x2 - x1, y2 - y1], [height, width], 0, img_dims))

                        else:
                            nonempty += 1
                            populated_tiles.append((file_path, group, rle, [x1, y1, x2 - x1, y2 - y1], [height, width], 1, img_dims))

            len_sample = len(populated_tiles)
            num_empty_tiles_to_sample = int(len_sample * self.empty_tile_pct)
            num_pos_tiles_to_sample = int(len_sample * (1 - self.empty_tile_pct))

            empty_idxs_to_sample = np.random.choice(len(empty_tiles), min(num_empty_tiles_to_sample, len(empty_tiles)), replace=False)
            pos_idxs_to_sample = np.random.choice(len(populated_tiles), min(num_pos_tiles_to_sample, len(populated_tiles)), replace=False)

            neg_samples = list(map(empty_tiles.__getitem__, empty_idxs_to_sample))
            pos_samples = list(map(populated_tiles.__getitem__, pos_idxs_to_sample))

            new_samples = pos_samples + neg_samples

            self.samples = new_samples
            if self.empty_tile_pct == 0.0:
                logger.info('Dropped {} empty tiles.'.format(empty))
            logger.info('Dataset contains {} empty and {} non-empty tiles.'.format(len(neg_samples), len(pos_samples)))
            logger.info(self.value_dict)

    def check_value(self, group, px_max, px_min):
        if group in self.value_dict:
            self.value_dict[group][0] = max(px_max, self.value_dict[group][0])
            self.value_dict[group][1] = min(px_max, self.value_dict[group][1])
        else:
            self.value_dict[group] = [px_max, px_min]

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns a sample of data

        """
        # Grab the sample from the sample list
        img_fpath, group, rle, bbox, original_img_size, target, img_dims = self.samples[idx]

        # If the user points us to a cache directory, generate the file paths to the cached imagery
        cache_file_img = None
        cache_file_mask = None
        if self.cache_dir is not None:
            cache_file_img = os.path.join(self.cache_dir, img_fpath.split('/')[-3], os.path.basename(img_fpath).split('.')[
                0] + f'_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.png')
            cache_file_mask = os.path.join(self.cache_dir, img_fpath.split('/')[-3],
                                       os.path.basename(img_fpath).split('.')[
                                           0] + f'_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_mask.png')

        # If the cached image exists, load it. Otherwise, read the window from the full scene
        if cache_file_img and os.path.exists(cache_file_img):
            img = np.array(Image.open(cache_file_img))
        else:
            img = ropen(img_fpath).read(1, window=rasterio.windows.Window(*bbox) if len(bbox) > 0 else None)

            if len(original_img_size) == 0:
                original_img_size = img.shape

            if img.ndim == 3:
                img = np.mean(img, axis=2)

            max_value = self.value_dict[group][0]
            min_value = self.value_dict[group][1]
            # If we read the window from the full scene, compress the dynamic range to UINT8
            img = (img - min_value) / ((max_value - min_value) + 1e-7)
            img[img>1] = 1
            img *= 255.0
            img = img.astype(np.uint8)

        # If the cached mask exists, load it. Otherwise, decode the rle and grab the window
        if cache_file_mask and os.path.exists(cache_file_mask):
            mask = np.array(Image.open(cache_file_mask))
        else:
            mask = rle_decode(rle, img_shape=original_img_size)

            if len(bbox) > 0:
                x, y, w, h = bbox
                mask = mask[y:y + h, x:x + w]

        # If the user requested use of a cache dir and the images don't exist on-disk, write them.
        if self.cache_dir is not None:
            os.makedirs(os.path.dirname(cache_file_img), exist_ok=True)
            if not os.path.exists(cache_file_img):
                im = Image.fromarray(img)
                im.save(cache_file_img)

            if not os.path.exists(cache_file_mask):
                im = Image.fromarray(mask)
                im.save(cache_file_mask)

        # Transform the image and mask using Albumentations
        if self.transforms:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        # The augmentations may scale the mask to the range 0-1, with 1's becoming ~0.0039 (1/255).
        # Make sure we overwrite this in any case
        mask[mask > 0] = 1

        # Return the sample
        return img, mask, img_fpath, bbox, original_img_size, target

    def __len__(self) -> int:
        return len(self.samples)


class SubmitTiledDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            df_data: pd.DataFrame,
            transforms=None,
            tile_size=None,
            overlap_pct: float = 0.2,
            cache_dir: str = None
    ):
        """
        Generates an in-memory tiled dataset using a sliding window approach (common in geospatial processing communities)

        Args:

            path_img_dir (str): Path to root directory containing data.
            transforms (torchvision.transforms.Transform): Albumentation transforms.
            tile_size (Optional[List[int]]): List describing the tile size. If None, tiling is not performed and the
                full scene is utilized.
            overlap_pct (float): Percentage of tile size to use as overlap between tiles. This effectively
                controls the "stride" of the window as it's slid across the full scene.
            cache_dir (str): Cache dir to read/write to. If None, no cache directory is utilized.
        """
        self.data = df_data
        self.transforms = transforms
        self.samples = []
        self.tile_size = np.array(tile_size) if tile_size is not None else None
        self.overlap_pct = overlap_pct
        self.cache_dir = cache_dir
        self.value_dict = dict()
        path_images = []
        for _, row in tqdm(self.data.iterrows()):
            p_img = row['img_path']
            with rasterio.open(p_img) as reader:
                width, height = reader.width, reader.height
                img = reader.read()
                px_max, px_min = img.max(), img.min()
                self.check_value(row['group'], px_max, px_min)

            if not os.path.isfile(p_img):
                continue
            self.samples.append((p_img, row['group'], [], [], 1, [width, height]))

        if self.tile_size is not None:
            empty = 0
            nonempty = 0

            empty_tiles = []
            populated_tiles = []

            for file_path, group, _, _, _, img_dims in tqdm(self.samples, total=len(self.samples), desc='Generating tiles'):
                width, height = img_dims

                min_overlap = float(overlap_pct) * 0.01
                max_stride = self.tile_size * (1.0 - min_overlap)
                num_patches = np.ceil(np.array([height, width]) / max_stride).astype(np.int64)
                # compute the cutoff points for the x-y dimensions
                starts = [np.int64(np.linspace(0, width - self.tile_size[1], num_patches[1])),
                          np.int64(np.linspace(0, height - self.tile_size[0], num_patches[0]))]
                stops = [starts[0] + self.tile_size[0], starts[1] + self.tile_size[1]]
                for y1, y2 in zip(starts[1], stops[1]):
                    for x1, x2 in zip(starts[0], stops[0]):

                        populated_tiles.append((file_path, group, [x1, y1, x2 - x1, y2 - y1], [height, width], 1, img_dims))


            pos_idxs_to_sample = range(len(populated_tiles))


            self.samples = list(map(populated_tiles.__getitem__, pos_idxs_to_sample))

    def check_value(self, group, px_max, px_min):
        if group in self.value_dict:
            self.value_dict[group][0] = max(px_max, self.value_dict[group][0])
            self.value_dict[group][1] = min(px_max, self.value_dict[group][1])
        else:
            self.value_dict[group] = [px_max, px_min]

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns a sample of data

        """
        # Grab the sample from the sample list
        img_fpath, group, bbox, original_img_size, target, img_dims = self.samples[idx]


        img = ropen(img_fpath).read(1, window=rasterio.windows.Window(*bbox) if len(bbox) > 0 else None)

        if len(original_img_size) == 0:
            original_img_size = img.shape

        if img.ndim == 3:
            img = np.mean(img, axis=2)

        # If we read the window from the full scene, compress the dynamic range to UINT8
        # TODO: Save full scene statistics and compress relative to those, rather than tile level statistics
        max_value = self.value_dict[group][0]
        min_value = self.value_dict[group][1]
        img = (img - min_value) / ((max_value - min_value) + 1e-7)
        img[img > 1] = 1
        # img *= 255.0
        img = img.astype(np.float32)

        # Transform the image and mask using Albumentations
        if self.transforms:
            transformed = self.transforms(image=img)
            img = transformed['image']

        # The augmentations may scale the mask to the range 0-1, with 1's becoming ~0.0039 (1/255).
        # Make sure we overwrite this in any case

        # Return the sample
        return img, img_fpath, bbox, original_img_size, target

    def __len__(self) -> int:
        return len(self.samples)
