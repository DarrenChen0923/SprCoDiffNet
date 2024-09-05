import re
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.cli import get_parser
from data.get_data import download_data
from data.get_data import extract_data
from data.SpringbackDataset import SpringbackDataset
from models.diffusion.train_noise_model import train_denoise_model
from models.diffusion.evaluation_noise_model import evaluate_model
from models.diffusion.generate_negative_samples import generate_negative_samples
from models.diffusion.noisy_model import DenoiseModel


# cml arguments
parser = get_parser()
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-----------------------Parameters-------------------------------#
# set download url and output dir
drive_url = "https://drive.google.com/file/d/1GMyZ5lG-NGtu5Qb9bPCualocWUlyw9YG"
output_zip = args.project_root+ "/raw_data.zip"
output_dir = args.project_root+ "/data/raw/"
grid_size = str(args.grid) +"mm" 
neagtive_samples_output_dir = output_dir + "/negative_samples_" + grid_size +"/"
epochs = 1000
model_path = args.project_root + f'/trained_models/{args.load_model}'
generate_nagetive_samples_times = 10

#-----------------------Diffusion Model-------------------------------#
#-----------------------Pre-trained Model-------------------------------#


def main():
    
    #-----------------------Download Data-------------------------------#
    # download_data(drive_url, output_zip) # download file
    # extract_data(output_zip, output_dir) # extract zip file
   

    #-----------------------Data Load Model-------------------------------#
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # read data
    dataset = SpringbackDataset(data_dir=output_dir, grid_size=grid_size, transform=transform)
    
    item_name = []
    images = []
    labels = []
    for index in range(len(dataset.image_files)):
        item_name.append(dataset.image_files[index])
        image, label = dataset.__getitem__(index)
        images.append(image)
        labels.append(label)
    
    item_index = [re.findall(r'\d+', filename)[0] for filename in item_name]

    # build data_loader
    batch_size = 64
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    #-----------------------Train Model-------------------------------#
    if torch.cuda.is_available():
        model = DenoiseModel().cuda()
    else:
        model = DenoiseModel()
 
    # train_and_evluation_model(model,data_loader)

    
    #-----------------------Generate Negative Sample -------------------------------#
    model.load_state_dict(torch.load(model_path))
    generate_negative_sample(model,images,item_index,generate_nagetive_samples_times)
  
    

def train_and_evluation_model(model,data_loader):
    # train model
    train_denoise_model(model, data_loader, epochs)
    #evaluate model
    evaluate_model(model,data_loader)
    os.makedirs("trained_models",exist_ok=True)
    torch.save(model.state_dict(), f'trained_models/SprCoDiffNet_{args.grid}mm.pth')

    
def generate_negative_sample(model,images,item_index,generate_nagetive_samples_times):
        #generate negative samples
        generate_negative_samples(model,images,neagtive_samples_output_dir,item_index,generate_nagetive_samples_times)

if __name__ == "__main__":
    main()
