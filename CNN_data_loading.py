import yaml
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config('config.yaml')

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset_train = ImageFolder(config['data']['train_dir'], transform=transform)

