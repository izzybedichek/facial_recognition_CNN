import yaml
from torchvision.datasets import ImageFolder

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config('config.yaml')

emily_path = "/Users/shuang/Desktop/0_Emily/3_Fall25/ML/Project 2"
izzy_path = ""
dataset_train = ImageFolder(emily_path + '/data/train')
dataset_test = ImageFolder(emily_path + '/data/test')

print(f"Train dataset size: {len(dataset_train)}")
print(f"Test dataset size: {len(dataset_test)}")
