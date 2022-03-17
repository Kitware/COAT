# This file is part of COAT, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE file or
# https://github.com/Kitware/COAT/blob/master/LICENSE for details.

import torch
from torch import nn
import torch.nn.functional as F

class SoftmaxLoss(nn.Module):
    def __init__(self, cfg):
        super(SoftmaxLoss, self).__init__()

        self.feat_dim = cfg.MODEL.EMBEDDING_DIM
        self.num_classes = cfg.MODEL.LOSS.LUT_SIZE

        self.bottleneck = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.feat_dim, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, inputs, labels):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert inputs.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        target = labels.clone()
        target[target >= self.num_classes] = 5554

        feat = self.bottleneck(inputs)
        score = self.classifier(feat)
        loss = F.cross_entropy(score, target, ignore_index=5554)

        return loss


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

