import yaml
import torch
import numpy as np

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data.sampler import WeightedRandomSampler

from mini_helper_functions import get_mean_std

print("Starting CNN_data_loading.py")

# Loading config
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config('config.yaml')

if config["device"] == "mps":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

elif config['device'] == "cuda":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(config["device"])

# Data Transformations
"""
ToTensor: 
- converts an image from (height x width x channels) 
- to (C x H x W) and 
- normalizes pixel values to [0, 1]
- converts image to pyTorch tensor"""

# calculating ahead for Normalization
generic_transform = v2.Compose([
    v2.Grayscale(num_output_channels=1),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

generic_dataset = ImageFolder(config['data']['train_dir_emily'],
                             transform = generic_transform)

generic_loader = DataLoader(generic_dataset,
                          batch_size = config["data"]["batch_size"])

mean, std = get_mean_std(generic_loader)

# Preprocessing steps
train_transform = v2.Compose([
    v2.Grayscale(num_output_channels=1), # preprocessing
    v2.RandomHorizontalFlip(), # augmenting
    v2.ColorJitter(brightness=(0.5, 1.5), contrast=(0.8, 1.2)), # brightness and contrast augmentation
    v2.ToImage(), # preprocessing
    v2.ToDtype(torch.float32, scale=True), # preprocessing
    v2.Normalize(mean=mean, std=std), # preprocessing
])

val_transform = transforms.Compose([ # only preprocessing
    v2.Grayscale(num_output_channels=1),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=mean, std=std),
])

# Applying preprocessing to dataset
dataset_train_val = ImageFolder("/Users/huang/Documents/NEU_Code/data/train", transform=train_transform)
dataset_test  = ImageFolder("/Users/huang/Documents/NEU_Code/data/test", transform=val_transform)

dataset_train, dataset_val = torch.utils.data.random_split(dataset_train_val, [.9, .1])

# Verifying we have the whole dataset
print(f"Train dataset size: {len(dataset_train)}")
print(f"Test dataset size: {len(dataset_test)}")

# Weighted random sampling for testing data
train_indices = dataset_train.indices
train_targets = [dataset_train_val.targets[i] for i in train_indices]

class_counts = np.bincount(train_targets)

#print(class_counts)

class_weights = 1.0 / class_counts

#print(class_weights)

sample_weights = [class_weights[t] for t in train_targets]

sampler = WeightedRandomSampler(weights = sample_weights,
                                num_samples = len(sample_weights),
                                replacement=True)

# Loading the dataset using PyTorch
train_loader = DataLoader(dataset_train,
                          sampler = sampler,
                          batch_size = config["data"]["batch_size"],
                          shuffle = False) # Shuffle OR sampler, not both

val_loader = DataLoader(dataset_val,
                        batch_size = config["data"]["batch_size"],
                        shuffle=False)


# Verifying all the data got into the loader
#print('train size:', len(train_loader))

# Verifying the batches and labels
images, labels = next(iter(train_loader))
#print('Batch shape:', images.shape, 'Labels shape:', labels.shape)



