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
    #v2.RandomHorizontalFlip(p=0.5),
    #v2.RandomRotation(20),
    #v2.RandomChoice(random_choices),
    #v2.CenterCrop(43),
    v2.ColorJitter(brightness=(0.5, 1.5), contrast=(0.8, 1.2)), # brightness and contrast augmentation
    v2.ToImage(), # preprocessing
    v2.ToDtype(torch.float32, scale=True), # preprocessing
    v2.Normalize(mean=mean, std=std), # preprocessing
    #v2.RandomErasing(p=0.2),
])

#use for validation and test dataset (no data augmentation)
val_transform = transforms.Compose([ # only preprocessing
    v2.Grayscale(num_output_channels=1),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=mean, std=std),
])

# Wrapper class
class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.subset)
    
    @property
    def targets(self):
        return [self.subset.dataset.targets[i] for i in self.subset.indices]

# Load in datasets
# train dataset has no preprocessing applied yet since we need to split the validation from it first
base_train_dataset = ImageFolder("/Users/huang/Documents/NEU_Code/data/train")
dataset_test  = ImageFolder("/Users/huang/Documents/NEU_Code/data/test", transform=val_transform)

# Split base_train to validation and train datasets
dataset_train, dataset_val = torch.utils.data.random_split(base_train_dataset, [.9, .1])

# Apply transforms to train and validation
dataset_train = TransformDataset(dataset_train, transform=train_transform)
dataset_val = TransformDataset(dataset_val, transform=val_transform)

# Verifying we have the whole dataset
print(f"Train dataset size: {len(dataset_train)}")
print(f"Validation dataset size: {len(dataset_val)}")
print(f"Test dataset size: {len(dataset_test)}")

# If weighted sampling is true in config > load training data with weighted sampling
if config["data"].get("weighted_sampling", False):
    print("Using weighted random sampling")
    # Weighted random sampling for testing data
    train_targets = dataset_train.targets

    class_counts = np.bincount(train_targets)
    class_weights = 1.0 / class_counts

    class_weights_tensor = torch.tensor( # use this in loss OR use weights= sample_weights, using both causes obsession with class 1
    class_weights,
    dtype=torch.float32,
    device=device
    )

    sample_weights = [class_weights[t] for t in train_targets]

    sampler = WeightedRandomSampler(weights = sample_weights,
                                    num_samples = len(sample_weights),
                                    replacement=True)

    # Loading the dataset using PyTorch
    train_loader = DataLoader(dataset_train,
                            sampler = sampler,
                            batch_size = config["data"]["batch_size"],
                            shuffle = False) # Shuffle OR sampler, not both
else:
     print("Using non-weighted training dataset")
     # Loading the dataset with regular random sampling with Shuffle
     train_loader = DataLoader(dataset_train,
                              batch_size = config["data"]["batch_size"],
                              shuffle = True)

# Validation dataset
val_loader = DataLoader(dataset_val,
                        batch_size = config["data"]["batch_size"],
                        shuffle=False)

# Test dataset
test_loader = DataLoader(
    dataset_test, 
    batch_size=config["data"]["batch_size"],
    shuffle=False  # Don't shuffle test data
)


# Verifying all the data got into the loader
#print('train size:', len(train_loader))

# Verifying the batches and labels
images, labels = next(iter(train_loader))
#print('Batch shape:', images.shape, 'Labels shape:', labels.shape)



