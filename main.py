import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt


from utils.cli import get_parser

from data.get_data import download_data
from data.get_data import extract_data
from data.SpringbackDataset import SpringbackDataset
from models.diffusion.unet import UNet
from models.diffusion.noisy_model import DenoiseModel
from models.diffusion.noisy import add_noise
from utils.utils import calculate_psnr
from utils.utils import calculate_ssim
from utils.utils import to_numpy

# cml arguments
parser = get_parser()
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#-----------------------Parameters-------------------------------#
# set download url and output dir
drive_url = "https://drive.google.com/file/d/1GMyZ5lG-NGtu5Qb9bPCualocWUlyw9YG"

# drive_url = "https://drive.google.com/file/d/1GrSz99zTtJSzKyFBNA6_NyS_af5rGplp"
output_zip = args.project_root+ "/raw_data.zip"
output_dir = args.project_root+ "/data/raw/"
grid_size = str(args.grid) +"mm" 
epochs = 1000



#-----------------------Diffusion Model-------------------------------#

def test_unet():
    model = UNet(in_channels=3, out_channels=3)
    
    # test different input image size
    sizes = [(1, 3, 15, 15), (1, 3, 30, 30), (1, 3, 45, 45), (1, 3, 60, 60)]
    
    for size in sizes:
        #generate random input accroding to the input image size
        x = torch.randn(size)  
        preds = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {preds.shape}")
        assert preds.shape == x.shape, f"Output shape does not match input shape for size {size}!"
    
    print("UNet forward pass test passed for all input sizes!")


def train_denoise_model(model, dataloader, num_epochs=epochs):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        for images, _ in dataloader:
            images = images.to(device)
          
            # random select noise_level
            noise_level = torch.rand(images.size(-1)).to(device)
            noisy_images = add_noise(images, noise_level)
            
            predicted_noise = model(noisy_images, noise_level)
            loss = criterion(predicted_noise, noisy_images - images)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")





#-----------------------Pre-trained Model-------------------------------#

def main():
    
    #-----------------------Data Load Model-------------------------------#
    download_data(drive_url, output_zip) # download file
    extract_data(output_zip, output_dir) # extract zip file
   

    #-----------------------Data Load Model-------------------------------#
    # test_unet()
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # read data
    
    dataset = SpringbackDataset(data_dir=output_dir, grid_size=grid_size, transform=transform)

    
    # build data_loader
    batch_size = 16
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # test dataloader
    for images, labels in data_loader:
        print(f"Images batch shape: {images.shape}")
        print(f"Labels: {labels}")
        break
    
    if len(dataset) > 0:
        print(f'DataLoader successfully loaded the data. Dataset length:{len(dataset)}')
    else:
        print("DataLoader failed to load any data.")


    #-----------------------Diffusion Model-------------------------------#
    # test_unet()
    # init model
    model = DenoiseModel().to(device)

    # train model
    train_denoise_model(model, data_loader, num_epochs=epochs)

    # Test on a single image
    model.eval()
    with torch.no_grad():
        original_img = images[0].unsqueeze(0).to(device)  # Example image from batch
        noise_level = torch.tensor([0.5]).view(1, 1, 1, 1).to(device)
        
        noisy_img = add_noise(original_img, noise_level)
        predicted_noise = model(noisy_img, noise_level)
        denoised_img = noisy_img - predicted_noise
        
        # Calculate PSNR and SSIM
        psnr_value = calculate_psnr(original_img, denoised_img)
        print(f"PSNR: {psnr_value} dB")
        ssim_value = calculate_ssim(to_numpy(original_img), to_numpy(denoised_img))
        print(f"SSIM: {ssim_value}")
        
        # Visualization
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(to_numpy(original_img))

        plt.subplot(1, 3, 2)
        plt.title("Noisy Image")
        plt.imshow(to_numpy(noisy_img))

        plt.subplot(1, 3, 3)
        plt.title("Denoised Image")
        plt.imshow(to_numpy(denoised_img))

        plt.show()
    


if __name__ == "__main__":
    main()



# from torchviz import make_dot
# import torch

# def visualize_unet(model, input_size=(1, 3, 60, 60)):
#     x = torch.randn(input_size)
#     y = model(x)
#     return make_dot(y, params=dict(model.named_parameters()))

# if __name__ == "__main__":
#     model = UNet(in_channels=3, out_channels=3)
#     unet_viz = visualize_unet(model)
#     unet_viz.render("unet_architecture", format="jpg")
