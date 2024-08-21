import torch

def add_noise(original_img, noise_level):
    """
    在给定的噪声水平下，为图像添加噪声。
    
    Args:
    - original_img (Tensor): 原始图像张量，形状为 (B, C, H, W)。
    - noise_level (float): 噪声的比例。

    Returns:
    - noisy_img (Tensor): 添加噪声后的图像。
    """
    # print("Noise Level:",noise_level.shape)
    # print("original_img shape:",original_img.shape)
    noise = torch.randn_like(original_img)
    # print("Noise shape:",noise.shape)
    noisy_img = original_img * (1 - noise_level) + noise * noise_level
    return noisy_img
