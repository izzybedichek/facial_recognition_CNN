import yaml
import torch
import numpy as np

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data.sampler import WeightedRandomSampler

from mini_helper_functions import get_mean_std

# Loading config
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config('config.yaml')

# making it run on Izzy's mac faster
if config["device"] == "mps":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
else:
    device = torch.device(config["device"])


# calculating ahead for Normalization
generic_transform = v2.Compose([
    v2.Grayscale(num_output_channels=1),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

generic_dataset = ImageFolder(config['data']['train_dir_izzy'],
                             transform = generic_transform)

generic_loader = DataLoader(generic_dataset,
                          batch_size = config["data"]["batch_size"])

mean, std = get_mean_std(generic_loader)

# preprocessing (works better WITHOUT augmentation, but leaving commented out for display purposes)
train_transform = v2.Compose([
    v2.Grayscale(num_output_channels=1), # double-ensuring everything is 1 channel
    #v2.RandomHorizontalFlip(p=0.5),
    #v2.RandomRotation(20),
    v2.ToImage(), # even though inputs are already images, this converts them into a tensor
    v2.ToDtype(torch.float32, scale=True), # scales values between 0 and 1
    v2.Normalize(mean=[mean], std=[std]), # normalizes values
    #v2.RandomErasing(p=0.2),
])

val_transform = transforms.Compose([ # only preprocessing
    v2.Grayscale(num_output_channels=1),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=mean, std=std),
])


### -----more preprocessing----- ###

# applying preprocessing to dataset
dataset_train_val = ImageFolder(config['data']['train_dir_izzy'], transform=train_transform)
dataset_test  = ImageFolder(config['data']['test_dir_izzy'], transform=val_transform)

# splitting training into training and validation
dataset_train, dataset_val = torch.utils.data.random_split(dataset_train_val, [.8, .2])


# Verifying we have the whole dataset
#print(f"Train dataset size: {len(dataset_train)}")
#print(f"Test dataset size: {len(dataset_test)}")

###------ weighted random sampling for training data -----###
train_indices = dataset_train.indices
train_targets = [dataset_train_val.targets[i] for i in train_indices]

class_counts = np.bincount(train_targets)

#print(class_counts)

class_weights = 1.0 / class_counts

class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=device
)

#print(class_weights)

sample_weights = [class_weights[t] for t in train_targets]

# weighted random sampling into 'sampler'
sampler = WeightedRandomSampler(weights = sample_weights,
                                num_samples = len(sample_weights),
                                replacement=True)


# train loader with weighted random sampling using 'sample'
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



