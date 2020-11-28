from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
"""
Custom transformations for data augmentation. 
"""

class CustomTransforms():

    crop_size = (420, 560)

    def __call__(self, img, label):

        # brightness & saturation
        img = TF.adjust_saturation(img, random.uniform(0.5, 1.5))
        img = TF.adjust_brightness(img, random.uniform(0.5, 1.5))

        # random rotation
        img, label = self.rotate(img, label)

        # random crop.
        img, label = self.randomcrop(img, label)
        return img, label

    def randomcrop(self, image, label):
        width, height = 640, 480
        outh, outw = self.crop_size

        crop = transforms.RandomCrop(self.crop_size)
        i, j, h, w = crop.get_params(image, self.crop_size)

        label[:, 0] = (label[:, 0] - (j / width)) * width / outw
        label[:, 1] = (label[:, 1] - (i / height)) * height / outh
        return TF.crop(image, i, j, h, w), label

    def rotate(self, img, label):
        angle = random.randint(-12, 12)
        img = TF.rotate(img, angle)

        label[:,:] = label[:,:]-0.5

        theta = np.radians(-angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        label = np.matmul(R, label.T).T

        label[:,:] = label[:,:]+0.5
        return img, label

