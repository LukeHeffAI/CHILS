import hydra
import os
import sys

# Set the project root as the current directory where run.py is located.
project_root = os.path.abspath(os.path.dirname(__file__))
os.environ["PYTHONPATH"] = project_root  # This sets PYTHONPATH for child processes.
sys.path.insert(0, project_root)

dataset_name = "food-101"

from os.path import join
from omegaconf import DictConfig, OmegaConf
from src.train import train
from src.utils import filter_config, get_dict_hash
from src.simple_utils import load_pickle, dump_pickle
from src.data_utils import get_dataset
import torchvision.transforms as transforms

# Dataset configuration mapping
DATASET_CONFIG = {
    "cub": {"num_classes": 200, "base_task": "cub"},
    "dtd": {"num_classes": 47, "base_task": "dtd"},
    "pets": {"num_classes": 37, "base_task": "pets"},
    "places365": {"num_classes": 365, "base_task": "places365"},
    "food-101": {"num_classes": 101, "base_task": "food-101"},
    "imagenet": {"num_classes": 1000, "base_task": "imagenet"},
    "eurosat": {"num_classes": 10, "base_task": "eurosat"},
    "fruits360": {"num_classes": 131, "base_task": "fruits360"},
    "fashion-mnist": {"num_classes": 10, "base_task": "fashion-mnist"},
    "fashion1m": {"num_classes": 14, "base_task": "fashion1m"},
    "resisc45": {"num_classes": 45, "base_task": "resisc45"},
    "office31-amazon": {"num_classes": 31, "base_task": "office31-amazon"},
    "office31-dslr": {"num_classes": 31, "base_task": "office31-dslr"},
    "office31-webcam": {"num_classes": 31, "base_task": "office31-webcam"},
    "officehome-product": {"num_classes": 65, "base_task": "officehome-product"},
    "officehome-realworld": {"num_classes": 65, "base_task": "officehome-realworld"},
    "officehome-art": {"num_classes": 65, "base_task": "officehome-art"},
    "officehome-clipart": {"num_classes": 65, "base_task": "officehome-clipart"}
}

def update_dataset_config(config, dataset_name=dataset_name):
    """
    Update config with dataset-specific parameters.
    If the dataset is not in our known mapping, try to load it to determine classes.
    """
    # Ensure the config is mutable
    OmegaConf.set_struct(config, False)

    dataset_name = config.datamodule.target_dataset.lower()
    
    # Try to use predefined config first
    if dataset_name in DATASET_CONFIG:
        dataset_info = DATASET_CONFIG[dataset_name]
        # Set values safely using OmegaConf.update
        OmegaConf.update(config, "num_classes", dataset_info["num_classes"], merge=True)
        OmegaConf.update(config, "base_task", dataset_info["base_task"], merge=True)
        OmegaConf.update(config, "source_dataset", dataset_name, merge=True)
        
        # Also update the datamodule config
        OmegaConf.update(config.datamodule, "num_classes", dataset_info["num_classes"], merge=True)
        
        print(f"Using predefined configuration for {dataset_name}: {dataset_info['num_classes']} classes")
        return config
    
    # If not in predefined config, try to determine dynamically
    try:
        # Simple transform just for class detection
        dummy_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])
        
        # Load dataset to get number of classes
        dataset = get_dataset(
            data_dir=os.path.expanduser(config.data_dir), 
            dataset=dataset_name,
            train=False, 
            transform=dummy_transform
        )
        
        if hasattr(dataset, 'classes'):
            num_classes = len(dataset.classes)
        elif hasattr(dataset, 'class_to_idx'):
            num_classes = len(dataset.class_to_idx)
        else:
            # Fallback - try to infer from targets
            targets = getattr(dataset, 'targets', [])
            if targets:
                num_classes = max(targets) + 1
            else:
                num_classes = 1000  # Default fallback
                print(f"Warning: Could not determine number of classes for {dataset_name}. Using default: {num_classes}")
        
        # Set values safely using OmegaConf.update
        OmegaConf.update(config, "num_classes", num_classes, merge=True)
        OmegaConf.update(config, "base_task", dataset_name, merge=True)
        OmegaConf.update(config, "source_dataset", dataset_name, merge=True)
        
        # Also update the datamodule config
        OmegaConf.update(config.datamodule, "num_classes", num_classes, merge=True)
        
        # Add to known configurations for future use
        DATASET_CONFIG[dataset_name] = {
            "num_classes": num_classes,
            "base_task": dataset_name
        }
        
        print(f"Dynamically configured dataset {dataset_name} with {num_classes} classes")
        
    except Exception as e:
        print(f"Error determining dataset configuration for {dataset_name}: {e}")
        print("Using default configuration values.")
        
        # Set safe fallback values
        OmegaConf.update(config, "num_classes", 1000, merge=True)
        OmegaConf.update(config, "base_task", dataset_name, merge=True)
        OmegaConf.update(config, "source_dataset", dataset_name, merge=True)
        
        # Also update the datamodule config
        OmegaConf.update(config.datamodule, "num_classes", 1000, merge=True)
    
    # Reset struct flag to original state if needed
    OmegaConf.set_struct(config, True)
    return config

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    # Update config with dataset-specific parameters
    config = update_dataset_config(config)
    
    print(OmegaConf.to_yaml(config))
    # extract data and model experiment info to group runs
    group_dict = dict(filter_config(config.datamodule), **filter_config(config.models))
    # group_dict["name"] = get_class_name(config.datamodule._target_, "train")

    group_hash = get_dict_hash(group_dict)
    config.logger.group = group_hash
    print(group_dict)

    if not os.path.isdir(config.log_dir):
        os.mkdir(config.log_dir)

    hash_dict_fname = join(config.log_dir, "hash_dict.pkl")

    if os.path.isfile(hash_dict_fname):
        hash_dict = load_pickle(hash_dict_fname)
    else:
        hash_dict = dict()

    hash_dict[group_hash] = group_dict
    dump_pickle(hash_dict, hash_dict_fname)

    raw_path = join(config.log_dir, "raw")
    if not os.path.isdir(raw_path):
        os.mkdir(raw_path)

    # start training
    train(config)       # TODO: add transforms to L/14 model size in data

if __name__ == "__main__":
    main()