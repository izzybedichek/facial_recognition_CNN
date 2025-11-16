import yaml
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config('config.yaml')

# Data tranformations: 
# ToTensor: converts image from (height x width x channels) to (C x H x W) and normalizes pixel values to [0, 1]
#            converts image to pyTorch tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1), # making sure everything is grayscale
])

dataset_train = ImageFolder(config['data']['train_dir_izzy'], transform=transform)
dataset_test  = ImageFolder(config['data']['test_dir_izzy'], transform=transform)

print(f"Train dataset size: {len(dataset_train)}")
print(f"Test dataset size: {len(dataset_test)}")


batch_size = 32
train_loader = DataLoader(dataset_train, batch_size=config["data"]["batch_size"], shuffle=True)

print('train size:', len(train_loader))

# Look at sample of a batch
# Batch shape: (batch_size, number of channels, height, width) - should be [__, 3, 48, 48]
# Labels shape: (batch_size)
images, labels = next(iter(train_loader))
print('Batch shape:', images.shape, 'Labels shape:', labels.shape)

