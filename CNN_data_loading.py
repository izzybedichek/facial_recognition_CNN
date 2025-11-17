import yaml
import numpy as np

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.sampler import WeightedRandomSampler

# Loading config
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config('config.yaml')


"""Data tranformations:
ToTensor: converts image from (height x width x channels) to (C x H x W) and normalizes pixel values to [0, 1]
          converts image to pyTorch tensor"""

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1), # making sure everything is grayscale
])

# Applying transformations to dataset
dataset_train = ImageFolder(config['data']['train_dir_izzy'], transform=transform)
dataset_test  = ImageFolder(config['data']['test_dir_izzy'], transform=transform)

# Verifying we have the whole dataset
#print(f"Train dataset size: {len(dataset_train)}")
#print(f"Test dataset size: {len(dataset_test)}")

# Weighted random sampling for testing data
class_counts = np.bincount(dataset_train.targets)

#print(class_counts)

class_weights = 1.0 / class_counts

#print(class_weights)

sample_weights = [class_weights[t] for t in dataset_train.targets]

sampler = WeightedRandomSampler(weights = sample_weights,
                                num_samples = len(sample_weights),
                                replacement=True)

# Loading the dataset using PyTorch
train_loader = DataLoader(dataset_train,
                          sampler = sampler,
                          batch_size = config["data"]["batch_size"],
                          shuffle = False) # Shuffle OR sampler, not both

# Verifying all the data got into the loader
#print('train size:', len(train_loader))

# Verifying the batches and labels
images, labels = next(iter(train_loader))
#print('Batch shape:', images.shape, 'Labels shape:', labels.shape)



