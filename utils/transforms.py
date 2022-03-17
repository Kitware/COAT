# This file is part of COAT, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE file or
# https://github.com/Kitware/COAT/blob/master/LICENSE for details.

import random
import math
import torch
import numpy as np
from copy import deepcopy
from torchvision.transforms import functional as F

def mixup_data(images, alpha=0.8):
    if alpha > 0. and alpha < 1.:
        lam = random.uniform(alpha, 1)
    else:
        lam = 1.

    batch_size = len(images)
    min_x = 9999
    min_y = 9999
    for i in range(batch_size):
        min_x = min(min_x, images[i].shape[1])
        min_y = min(min_y, images[i].shape[2])

    shuffle_images = deepcopy(images)
    random.shuffle(shuffle_images)
    mixed_images = deepcopy(images)
    for i in range(batch_size):
        mixed_images[i][:, :min_x, :min_y] = lam * images[i][:, :min_x, :min_y] + (1 - lam) * shuffle_images[i][:, :min_x, :min_y]

    return mixed_images

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=2, length=100):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img, target):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img, target


class RandomErasing(object):
    '''
    https://github.com/zhunzhong07/CamStyle/blob/master/reid/utils/data/transforms.py
    '''
    def __init__(self, EPSILON=0.5, mean=[0.485, 0.456, 0.406]):
        self.EPSILON = EPSILON
        self.mean = mean

    def __call__(self, img, target):
        if random.uniform(0, 1) > self.EPSILON:
            return img, target

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(0.02, 0.2) * area
            aspect_ratio = random.uniform(0.3, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size()[2] and h <= img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]

                return img, target

        return img, target


class ToTensor:
    def __call__(self, image, target):
        # convert [0, 255] to [0, 1]
        image = F.to_tensor(image)
        return image, target


def build_transforms(cfg, is_train):
    transforms = []
    transforms.append(ToTensor())
    if is_train:
        transforms.append(RandomHorizontalFlip())
        if cfg.INPUT.IMAGE_CUTOUT:
            transforms.append(Cutout())
        if cfg.INPUT.IMAGE_ERASE:
            transforms.append(RandomErasing())

    return Compose(transforms)
