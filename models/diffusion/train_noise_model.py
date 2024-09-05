import torch.optim as optim
import torch.nn as nn
import torch
import torch.optim as optim
from models.diffusion.noisy import add_noise
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/mnist_experiment_1')

def train_denoise_model(model, dataloader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            writer.add_scalar('training loss', loss, epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    
    writer.close()

