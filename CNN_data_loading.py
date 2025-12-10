import yaml
import torch
import numpy as np

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data.sampler import WeightedRandomSampler

from mini_helper_functions import get_mean_std, TransformSubset

# loading config
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
###---- calculating ahead for normalization -----###
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

###---- preprocessing (works better WITHOUT augmentation, but leaving commented out for display purposes)---###
random_choices = [ # to make the model focus more on internal facial features over face outline
    v2.CenterCrop(43),
    v2.CenterCrop(38),
]

train_transform = v2.Compose([
    v2.Grayscale(num_output_channels=1), # double-ensuring everything is 1 channel
    #v2.RandomHorizontalFlip(p=0.5),
    #v2.RandomRotation(20),
    #v2.RandomChoice(random_choices),
    #v2.CenterCrop(43),
    #v2.Resize((48,48)), # for the random crops
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
# load dataset without transforms
dataset_train_val = ImageFolder(config['data']['train_dir_izzy'], transform=None)

# split indices (not the dataset itself)
train_size = int(0.8 * len(dataset_train_val))
val_size = len(dataset_train_val) - train_size
train_indices, val_indices = torch.utils.data.random_split(
    range(len(dataset_train_val)),
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # For reproducibility
)

# apply transforms to split datasets
dataset_train = TransformSubset(
    Subset(dataset_train_val, train_indices.indices),
    transform=train_transform
)

dataset_val = TransformSubset(
    Subset(dataset_train_val, val_indices.indices),
    transform=val_transform
)

# Load test dataset
dataset_test = ImageFolder(config['data']['test_dir_izzy'], transform=val_transform)


# Verifying we have the whole dataset
#print(f"Train dataset size: {len(dataset_train)}")
#print(f"Test dataset size: {len(dataset_test)}")

###------ weighted random sampling for training data -----###
train_indices_list = train_indices.indices  # This is from the random_split
train_targets = [dataset_train_val.targets[i] for i in train_indices_list]

class_counts = np.bincount(train_targets)

#print(class_counts)

class_weights = 1.0 / class_counts

class_weights_tensor = torch.tensor( # use this in loss OR use weights= sample_weights, using both causes obsession with class 1
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

test_loader = DataLoader(dataset_test, batch_size=config["data"]["batch_size"], shuffle=False)
###----- Verifying all the data got into the loader---###

#print('train size:', len(train_loader))

# Verifying the batches and labels
images, labels = next(iter(train_loader))
#print('Batch shape:', images.shape, 'Labels shape:', labels.shape)



