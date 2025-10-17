import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from matplotlib import pyplot as plt



class Loss:
    def __init__(self, priorWeight, appearanceWeight, shadingWeight, sparsityWeight, size=64):
        """
        priorWeight: weight for the prior loss.
        appearanceWeight: weight for the appearance loss.
        shadingWeight: weight for the shading loss.
        sparsityWeight: weight for the sparsity loss.
        size: image spatial size, default 64.
        """
        self.priorWeight = priorWeight
        self.appearanceWeight = appearanceWeight
        self.shadingWeight = shadingWeight
        self.sparsityWeight = sparsityWeight
        self.size = size

    def __call__(self, rgb, shade, spec, b, shading, mask, x):
        scale = torch.sum(shade * shading * mask, (1, 2)) / (torch.sum(shade * shade * mask, (1, 2)) + 1e-9)
        scaledShading = torch.reshape(scale, (-1, 1, 1)) * shade
        alpha = (shading - scaledShading) * mask
        priorLoss = torch.sum(b ** 2) * self.priorWeight / x.shape[0]

        originalImage = torch.clone(x)
        # originalImage[:, 0, :, :] += mean_pixel[0]
        # originalImage[:, 1, :, :] += mean_pixel[1]
        # originalImage[:, 2, :, :] += mean_pixel[2]

        # rgb = rgb + mean_pixel.reshape((1, 3, 1, 1))

        delta = ((originalImage - rgb) ** 2) * torch.reshape(mask, (-1, 1, self.size, self.size))
        # delta = (originalImage - rgb) ** 2
        appearanceLoss = torch.sum(delta ** 2 / (self.size * self.size)) * 255 * 255 * self.appearanceWeight / x.shape[0]
        # Matlab implementation has image in (0 - 255) so we scale appropriately
        shadingLoss = torch.sum(alpha ** 2) * self.shadingWeight / x.shape[0]
        # Paper mentions divide by size of mask but not in implementation
        # computing on spec sparsity loss after lightColor transformation

        sparsityLoss = torch.sum(spec) * self.sparsityWeight / x.shape[0]  # Change to 1e-7
        # print(f"priorLoss: {priorLoss}, appearanceLoss: {appearanceLoss}, shadingLoss: {shadingLoss}, sparsityLoss: {sparsityLoss}")
        """Return four loss terms: prior, appearance, shading and sparsity."""
        return priorLoss, appearanceLoss, shadingLoss, sparsityLoss