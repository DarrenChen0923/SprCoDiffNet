import torch
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import math

# PSNR Calculation
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(1.0 / math.sqrt(mse))
    return psnr

# SSIM Calculation
def calculate_ssim(img1, img2):
    img1_resized = resize(img1, (7, 7), anti_aliasing=True)
    img2_resized = resize(img2, (7, 7), anti_aliasing=True)

    # ssim_value = ssim(img1_resized, img2_resized, channel_axis=-1)
    return ssim(img1_resized, img2_resized, data_range=1, channel_axis=-1)
    # img1 = img1.squeeze()
    # img2 = img2.squeeze()
    # return ssim(img1, img2, multichannel=True, data_range=img1.max() - img1.min())

# Convert tensor to numpy for visualization
def to_numpy(tensor):
    return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
