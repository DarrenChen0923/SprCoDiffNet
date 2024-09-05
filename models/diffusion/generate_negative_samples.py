import os
import torch
from models.diffusion.noisy import add_noise
from torchvision.utils import save_image


def generate_negative_samples(model, images, output_dir, original_filenames,generate_nagetive_samples_times):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        for i in range(generate_nagetive_samples_times):
            for idx, image in enumerate(images):
                image = image.unsqueeze(0).to(device)
                noise_level = torch.tensor([0.5]).view(1, 1, 1, 1).to(device)
                noisy_image = add_noise(image, noise_level)
                
                predicted_noise = model(noisy_image, noise_level)
                negative_sample = noisy_image - predicted_noise
                
                os.makedirs(output_dir,exist_ok=True)
                save_path = os.path.join(output_dir, f"negative_{original_filenames[idx]}_{i+1}.jpg")
                save_image(negative_sample.squeeze(0).cpu(), save_path)
            
            print(f'Finish genearte nagative samples {i+1} times')
