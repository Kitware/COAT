# This file is part of COAT, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE file or
# https://github.com/Kitware/COAT/blob/master/LICENSE for details.

import random
import torch

class exchange_token:
    def __init__(self):
        pass

    def __call__(self, features, mask_box):
        b, hw, c = features.size()
        assert hw == 14*14
        new_idx, mask_x1, mask_x2, mask_y1, mask_y2 = mask_box
        features = features.view(b, 14, 14, c)
        features[:, mask_x1 : mask_x2, mask_y1 : mask_y2, :] = features[new_idx, mask_x1 : mask_x2, mask_y1 : mask_y2, :]
        features = features.view(b, hw, c)
        return features

class jigsaw_token:
    def __init__(self, shift=5, group=2, begin=1):
        self.shift = shift
        self.group = group
        self.begin = begin

    def __call__(self, features):
        batchsize = features.size(0)
        dim = features.size(2)

        num_tokens = features.size(1)
        if num_tokens == 196:
            self.group = 2
        elif num_tokens == 25:
            self.group = 5
        else:
            raise Exception("Jigsaw - Unwanted number of tokens")

        # Shift Operation
        feature_random = torch.cat([features[:, self.begin-1+self.shift:, :], features[:, self.begin-1:self.begin-1+self.shift, :]], dim=1)
        x = feature_random

        # Patch Shuffle Operation
        try:
            x = x.view(batchsize, self.group, -1, dim)
        except:
            raise Exception("Jigsaw - Unwanted number of groups")

        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, dim)

        return x

class get_mask_box:
    def __init__(self, shape='stripe', mask_size=2, mode='random_direct'):
        self.shape = shape
        self.mask_size = mask_size
        self.mode = mode

    def __call__(self, features):
        # Stripe mask
        if self.shape == 'stripe':
            if self.mode == 'horizontal':
                mask_box = self.hstripe(features, self.mask_size)
            elif self.mode == 'vertical':
                mask_box = self.vstripe(features, self.mask_size)
            elif self.mode == 'random_direction':
                if random.random() < 0.5:
                    mask_box = self.hstripe(features, self.mask_size)
                else:
                    mask_box = self.vstripe(features, self.mask_size)
            else:
                raise Exception("Unknown stripe mask mode name")
        # Square mask
        elif self.shape == 'square':
            if self.mode == 'random_size':
                self.mask_size = 4 if random.random() < 0.5 else 5
            mask_box = self.square(features, self.mask_size)
        # Random stripe/square mask
        elif self.shape == 'random':
            random_num = random.random()
            if random_num < 0.25:
                mask_box = self.hstripe(features, 2)
            elif random_num < 0.5 and random_num >= 0.25:
                mask_box = self.vstripe(features, 2)
            elif random_num < 0.75 and random_num >= 0.5:
                mask_box = self.square(features, 4)
            else:
                mask_box = self.square(features, 5)
        else:
            raise Exception("Unknown mask shape name")
        return mask_box

    def hstripe(self, features, mask_size):
        """
        """
        # horizontal stripe
        mask_x1 = 0
        mask_x2 = features.shape[2]
        y1_max = features.shape[3] - mask_size
        mask_y1 = torch.randint(y1_max, (1,))
        mask_y2 = mask_y1 + mask_size
        new_idx = torch.randperm(features.shape[0])
        mask_box = (new_idx, mask_x1, mask_x2, mask_y1, mask_y2)
        return mask_box

    def vstripe(self, features, mask_size):
        """
        """
        # vertical stripe
        mask_y1 = 0
        mask_y2 = features.shape[3]
        x1_max = features.shape[2] - mask_size
        mask_x1 = torch.randint(x1_max, (1,))
        mask_x2 = mask_x1 + mask_size
        new_idx = torch.randperm(features.shape[0])
        mask_box = (new_idx, mask_x1, mask_x2, mask_y1, mask_y2)
        return mask_box

    def square(self, features, mask_size):
        """
        """
        # square
        x1_max = features.shape[2] - mask_size
        y1_max = features.shape[3] - mask_size
        mask_x1 = torch.randint(x1_max, (1,))
        mask_y1 = torch.randint(y1_max, (1,))
        mask_x2 = mask_x1 + mask_size
        mask_y2 = mask_y1 + mask_size
        new_idx = torch.randperm(features.shape[0])
        mask_box = (new_idx, mask_x1, mask_x2, mask_y1, mask_y2)
        return mask_box


class exchange_patch:
    def __init__(self, shape='stripe', mask_size=2, mode='random_direct'):
        self.shape = shape
        self.mask_size = mask_size
        self.mode = mode

    def __call__(self, features):
        # Stripe mask
        if self.shape == 'stripe':
            if self.mode == 'horizontal':
                features = self.xpatch_hstripe(features, self.mask_size)
            elif self.mode == 'vertical':
                features = self.xpatch_vstripe(features, self.mask_size)
            elif self.mode == 'random_direction':
                if random.random() < 0.5:
                    features = self.xpatch_hstripe(features, self.mask_size)
                else:
                    features = self.xpatch_vstripe(features, self.mask_size)
            else:
                raise Exception("Unknown stripe mask mode name")
        # Square mask
        elif self.shape == 'square':
            if self.mode == 'random_size':
                self.mask_size = 4 if random.random() < 0.5 else 5
            features = self.xpatch_square(features, self.mask_size)
        # Random stripe/square mask
        elif self.shape == 'random':
            random_num = random.random()
            if random_num < 0.25:
                features = self.xpatch_hstripe(features, 2)
            elif random_num < 0.5 and random_num >= 0.25:
                features = self.xpatch_vstripe(features, 2)
            elif random_num < 0.75 and random_num >= 0.5:
                features = self.xpatch_square(features, 4)
            else:
                features = self.xpatch_square(features, 5)
        else:
            raise Exception("Unknown mask shape name")

        return features

    def xpatch_hstripe(self, features, mask_size):
        """
        """
        # horizontal stripe
        y1_max = features.shape[3] - mask_size
        num_masks = 1
        for i in range(num_masks):
            mask_y1 = torch.randint(y1_max, (1,))
            mask_y2 = mask_y1 + mask_size
            new_idx = torch.randperm(features.shape[0])
            features[:, :, :, mask_y1 : mask_y2] = features[new_idx, :, :, mask_y1 : mask_y2]
        return features


    def xpatch_vstripe(self, features, mask_size):
        """
        """
        # vertical stripe
        x1_max = features.shape[2] - mask_size
        num_masks = 1
        for i in range(num_masks):
            mask_x1 = torch.randint(x1_max, (1,))
            mask_x2 = mask_x1 + mask_size
            new_idx = torch.randperm(features.shape[0])
            features[:, :, mask_x1 : mask_x2, :] = features[new_idx, :, mask_x1 : mask_x2, :]
        return features


    def xpatch_square(self, features, mask_size):
        """
        """
        # square
        x1_max = features.shape[2] - mask_size
        y1_max = features.shape[3] - mask_size
        num_masks = 1
        for i in range(num_masks):
            mask_x1 = torch.randint(x1_max, (1,))
            mask_y1 = torch.randint(y1_max, (1,))
            mask_x2 = mask_x1 + mask_size
            mask_y2 = mask_y1 + mask_size
            new_idx = torch.randperm(features.shape[0])
            features[:, :, mask_x1 : mask_x2, mask_y1 : mask_y2] = features[new_idx, :, mask_x1 : mask_x2, mask_y1 : mask_y2]
        return features


class cutout_patch:
    def __init__(self, mask_size=2):
        self.mask_size = mask_size

    def __call__(self, features):
        if random.random() < 0.5:
            y1_max = features.shape[3] - self.mask_size
            num_masks = 1
            for i in range(num_masks):
                mask_y1 = torch.randint(y1_max, (features.shape[0],))
                mask_y2 = mask_y1 + self.mask_size
                for k in range(features.shape[0]):
                    features[k, :, :, mask_y1[k] : mask_y2[k]] = 0
        else:
            x1_max = features.shape[3] - self.mask_size
            num_masks = 1
            for i in range(num_masks):
                mask_x1 = torch.randint(x1_max, (features.shape[0],))
                mask_x2 = mask_x1 + self.mask_size
                for k in range(features.shape[0]):
                    features[k, :, mask_x1[k] : mask_x2[k], :] = 0

        return features


class erase_patch:
    def __init__(self, mask_size=2):
        self.mask_size = mask_size

    def __call__(self, features):
        std, mean = torch.std_mean(features.detach())
        dim = features.shape[1]
        if random.random() < 0.5:
            y1_max = features.shape[3] - self.mask_size
            num_masks = 1
            for i in range(num_masks):
                mask_y1 = torch.randint(y1_max, (features.shape[0],))
                mask_y2 = mask_y1 + self.mask_size
                for k in range(features.shape[0]):
                    features[k, :, :, mask_y1[k] : mask_y2[k]] = torch.normal(mean.repeat(dim,14,2), std.repeat(dim,14,2))
        else:
            x1_max = features.shape[3] - self.mask_size
            num_masks = 1
            for i in range(num_masks):
                mask_x1 = torch.randint(x1_max, (features.shape[0],))
                mask_x2 = mask_x1 + self.mask_size
                for k in range(features.shape[0]):
                    features[k, :, mask_x1[k] : mask_x2[k], :] = torch.normal(mean.repeat(dim,2,14), std.repeat(dim,2,14))

        return features

class mixup_patch:
    def __init__(self, mask_size=2):
        self.mask_size = mask_size

    def __call__(self, features):
        lam = random.uniform(0, 1)
        if random.random() < 0.5:
            y1_max = features.shape[3] - self.mask_size
            num_masks = 1
            for i in range(num_masks):
                mask_y1 = torch.randint(y1_max, (1,))
                mask_y2 = mask_y1 + self.mask_size
                new_idx = torch.randperm(features.shape[0])
                features[:, :, :, mask_y1 : mask_y2] = lam*features[:, :, :, mask_y1 : mask_y2] + (1-lam)*features[new_idx, :, :, mask_y1 : mask_y2]
        else:
            x1_max = features.shape[2] - self.mask_size
            num_masks = 1
            for i in range(num_masks):
                mask_x1 = torch.randint(x1_max, (1,))
                mask_x2 = mask_x1 + self.mask_size
                new_idx = torch.randperm(features.shape[0])
                features[:, :, mask_x1 : mask_x2, :] = lam*features[:, :, mask_x1 : mask_x2, :] + (1-lam)*features[new_idx, :, mask_x1 : mask_x2, :]

        return features


class jigsaw_patch:
    def __init__(self, shift=5, group=2):
        self.shift = shift
        self.group = group

    def __call__(self, features):
        batchsize = features.size(0)
        dim = features.size(1)
        features = features.view(batchsize, dim, -1)

        # Shift Operation
        feature_random = torch.cat([features[:, :, self.shift:], features[:, :, :self.shift]], dim=2)
        x = feature_random

        # Patch Shuffle Operation
        try:
            x = x.view(batchsize, dim, self.group, -1)
        except:
            x = torch.cat([x, x[:, -2:-1, :]], dim=1)
            x = x.view(batchsize, self.group, -1, dim)

        x = torch.transpose(x, 2, 3).contiguous()

        x = x.view(batchsize, dim, -1)
        x = x.view(batchsize, dim, 14, 14)

        return x

