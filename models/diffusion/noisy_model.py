import torch.nn as nn
from models.diffusion.unet import UNet

class DenoiseModel(nn.Module):
    def __init__(self):
        super(DenoiseModel, self).__init__()
        self.unet = UNet(in_channels=3, out_channels=3)
    
    def forward(self, x, noise_level):
        """
        通过UNet预测噪声。
        
        Args:
        - x (Tensor): 输入的图像张量。
        - noise_level (Tensor): 当前时间步的噪声水平。

        Returns:
        - predicted_noise (Tensor): 预测的噪声。
        """
        # 将噪声水平作为条件输入到UNet中
        predicted_noise = self.unet(x)
        return predicted_noise
