import yaml
from torchvision.datasets import ImageFolder

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config('config.yaml')


dataset = ImageFolder('data/train', transform=transform)

