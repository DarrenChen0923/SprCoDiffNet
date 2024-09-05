import torch
import numpy as np
from models.diffusion.noisy import add_noise
from torchvision.transforms import ToPILImage
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# 模型评估函数
def evaluate_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            
            noise_level = torch.tensor([0.5]).view(1, 1, 1, 1).to(device)
            noisy_images = add_noise(images, noise_level)
            predicted_noise = model(noisy_images, noise_level)
            denoised_images = noisy_images - predicted_noise
            
            for i in range(images.size(0)):
                original_img = images[i].cpu().numpy()
                denoised_img = denoised_images[i].cpu().numpy()

                psnr_value,ssim_value = calculate_psnr_ssim(original_img,denoised_img)
                psnr_values.append(psnr_value)
                
                ssim_values.append(ssim_value)
    
    print(f'Average PSNR: {np.mean(psnr_values):.2f}')
    print(f'Average SSIM: {np.mean(ssim_values):.4f}')

def calculate_psnr_ssim(original_image, generated_image):
    # 将张量转换为 uint8
    original_image_uint8 = convert_tensor_to_uint8(original_image)
    generated_image_uint8 = convert_tensor_to_uint8(generated_image)

    # 使用 ToPILImage 进行转换
    original_image_pil = ToPILImage()(original_image_uint8)
    generated_image_pil = ToPILImage()(generated_image_uint8)

    # 转换为 numpy 数组
    original_image_np = np.array(original_image_pil)
    generated_image_np = np.array(generated_image_pil)

    # 计算 PSNR
    psnr_value = peak_signal_noise_ratio(original_image_np, generated_image_np, data_range=255)

    # 动态确定 win_size
    min_dim = min(original_image_np.shape[:2])  # 获取图像最小的维度
    win_size = min(7, min_dim // 2 * 2 + 1)  # 使用最小的维度计算 win_size，确保它是奇数并且不超过图像大小

    # 计算 SSIM，设置 win_size 和 channel_axis
    ssim_value = ssim(original_image_np, generated_image_np, win_size=win_size, multichannel=True, channel_axis=-1)

    return psnr_value, ssim_value

def convert_tensor_to_uint8(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)  # 如果是 numpy 数组，转换为 PyTorch 张量

    tensor = tensor * 255  # 假设张量值在 [0, 1] 范围内，缩放到 [0, 255]
    tensor = torch.clamp(tensor, 0, 255)  # 将张量限制在 [0, 255] 范围内
    tensor = tensor.byte()  # 转换为 uint8 类型
    return tensor
