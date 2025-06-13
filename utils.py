import numpy as np

def binary_cross_entropy(pred, target):
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
    return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class SegmentationDataset(Dataset):
    """Dataset genérico para imagens e máscaras."""
    def __init__(self, img_paths, mask_paths, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        img = self.transform(img)
        mask = self.transform(mask)
        return img, mask

def dice_coef(pred, target, eps=1e-7):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean()

def bce_dice_loss(pred, target):
    bce = F.binary_cross_entropy(pred, target)
    dice = 1 - dice_coef(pred, target)
    return bce + dice

def compute_metrics(pred, mask):
    pred_bin = (pred > 0.5).float()
    tp = (pred_bin * mask).sum()
    tn = ((1 - pred_bin) * (1 - mask)).sum()
    fp = (pred_bin * (1 - mask)).sum()
    fn = ((1 - pred_bin) * mask).sum()

    precision = tp / (tp + fp + 1e-7)
    sensitivity = tp / (tp + fn + 1e-7)
    specificity = tn / (tn + fp + 1e-7)
    dice = dice_coef(pred_bin, mask)

    return {
        'precision': precision.item(),
        'sensitivity': sensitivity.item(),
        'specificity': specificity.item(),
        'dice': dice.item()
    }
