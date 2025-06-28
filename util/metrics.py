# util/metrics.py
import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def tensor_to_numpy(img_tensor):
    # Convert to (H, W) or (H, W, C) depending on shape
    img = img_tensor.detach().cpu().float().numpy()
    if img.shape[0] == 1:
        img = img[0]  # (H, W)
    else:
        img = np.transpose(img, (1, 2, 0))  # (H, W, C)
    return img

def compute_ssim(img1, img2):
    img1 = tensor_to_numpy(img1)
    img2 = tensor_to_numpy(img2)
    return compare_ssim(img1, img2, data_range=img2.max() - img2.min(), multichannel=(img1.ndim == 3))

def compute_psnr(img1, img2):
    img1 = tensor_to_numpy(img1)
    img2 = tensor_to_numpy(img2)
    return compare_psnr(img1, img2, data_range=img2.max() - img2.min())

def compute_mae(img1, img2):
    return torch.mean(torch.abs(img1 - img2)).item()

def compute_mse(img1, img2):
    return torch.mean((img1 - img2) ** 2).item()

def compute_rmse(img1, img2):
    return compute_mse(img1, img2) ** 0.5
