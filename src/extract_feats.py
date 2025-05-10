import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix, MeanMetric
import torch
import torch.optim.lr_scheduler as lr_sched
from torch.nn.functional import softmax, one_hot
import torchvision.transforms as transforms
from typing import List, Optional
from src.model_utils import *
import logging 
import wandb
import numpy as np
import os
from src.data_utils import get_dataset

log = logging.getLogger("app")

class EvalNet(pl.LightningModule):
    def __init__(
        self,
        arch: str = "ClipViTB32",
        retrain: bool = False,
        base_task: str = "food-101",
        pretrained: bool = True,
        target_dataset: List[str] = [],
        # classnames: Optional[List[str]] = None,
        work_dir: str = ".",
        max_epochs: int = 1,
        hash: Optional[str] = None,
        num_classes: int = None  # Added to accept num_classes from config
    ):
        super().__init__()

        # Store all parameters as attributes right at the beginning 
        self.arch = arch
        self.base_task = base_task.lower()
        self.num_classes = num_classes
        self.target_dataset = target_dataset  
        self.work_dir = work_dir
        self.hash = hash
        self.pretrained = pretrained  # Assigned before use
        self.retrain = retrain
        
        # Get class names for the dataset before initializing CLIP
        self.task_classes = self.get_dataset_classes(self.base_task)
        
        # # Log the model initialization mode
        # if not self.pretrained:
        #     log.info(f"Training a new {arch} model")
        # else:
        #     log.info(f"Only extracting the features of a pre-trained {arch} model")
            
        # Create model and pass task_classes attribute to it
        self.model = get_model(arch=arch, dataset=base_task, pretrained=pretrained, 
                              retrain=retrain, extract_features=True, work_dir=work_dir)
        
        # Make the task_classes accessible to the CLIP model
        self.model.task_classes = self.task_classes
        
        # Now initialize text features with proper class information
        self.model.init_text(base_task.lower())
        
        self.input_resolution = 224

        # Get the input resolution required by the model
        if 'ViT' in arch:
            # Extract the model resolution based on architecture
            bb_str = str(self.model.bb)
            if 'ViT-L/14@336px' in bb_str:
                self.input_resolution = 336

    def get_dataset_classes(self, dataset_name):
        """
        Get class names for a dataset
        """
        dataset_name = dataset_name.lower()
        try:
            # Create a simple transform just to load the dataset
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor()
            ])
            
            # Use data_dir from the config, or fallback to a default path
            data_dir = os.path.expanduser("~/Documents/GitHub/data")
            
            # Get dataset to extract class names
            dataset = get_dataset(data_dir=data_dir, dataset=dataset_name, 
                                 train=False, transform=transform)
            
            # Extract class names from the dataset
            if hasattr(dataset, 'classes'):
                return dataset.classes
            elif hasattr(dataset, 'class_to_idx'):
                # Sort class names by index
                class_to_idx = dataset.class_to_idx
                class_names = [""] * len(class_to_idx)
                for name, idx in class_to_idx.items():
                    class_names[idx] = name
                return class_names
            else:
                # Fallback for datasets without explicit class names
                if self.num_classes is not None:
                    # Use numbered class names as a fallback
                    return [f"class_{i}" for i in range(self.num_classes)]
                else:
                    # Last resort fallback
                    return ["unknown_class"]
                
        except Exception as e:
            print(f"Error determining class names for {dataset_name}: {e}")
            # Return fallback class names based on num_classes if available
            if self.num_classes is not None:
                return [f"class_{i}" for i in range(self.num_classes)]
            else:
                # Default fallback
                return ["unknown_class"]

    def forward(self, x):
        # Ensure input is resized to the correct resolution
        if x.shape[-1] != self.input_resolution or x.shape[-2] != self.input_resolution:
            # Create a resize transform
            resize = transforms.Resize((self.input_resolution, self.input_resolution))
            x = resize(x)
        return self.model(x)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, _ = batch[:2]
        # Ensure input is resized to the correct resolution
        if x.shape[-1] != self.input_resolution or x.shape[-2] != self.input_resolution:
            # Create a resize transform
            resize = transforms.Resize((self.input_resolution, self.input_resolution))
            x = resize(x)
        return self.model(x)

    def process_batch(self, batch, stage="train", dataloader_idx=0):
        x, y, idx = batch[:3]
        
        # Ensure input is resized to the correct resolution
        if x.shape[-1] != self.input_resolution or x.shape[-2] != self.input_resolution:
            # Create a resize transform
            resize = transforms.Resize((self.input_resolution, self.input_resolution))
            x = resize(x)

        output = self.forward(x)
        logits = output["logits"]
        features = torch.flatten(output["features"],1)
        
        _, pred_idx = torch.max(logits, dim=1)

        return  y, pred_idx, idx, features


    def training_step(self, batch, batch_idx: int):
        _ = self.process_batch(batch, "pred")
        
        return None

        # pass

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        # Process the batch and get outputs as before:
        outputs = self.process_batch(batch, "pred", dataloader_idx)
        
        # Ensure self.test_outputs exists and append the outputs:
        if not hasattr(self, "test_outputs"):
            self.test_outputs = []
        self.test_outputs.append(outputs)
        
        return outputs


    def on_test_epoch_end(self):
        # Retrieve the stored outputs:
        outputs_list = self.test_outputs if hasattr(self, "test_outputs") else []

        # If nothing was stored, you might just exit:
        if not outputs_list:
            return

        # Process the stored outputs:
        labels = torch.cat([x[0] for x in outputs_list])
        outputs = torch.cat([x[1] for x in outputs_list])
        idx = torch.cat([x[2] for x in outputs_list])
        features = torch.cat([x[3] for x in outputs_list])

        # Create directory if it doesn't exist:
        out_dir = os.path.join(self.work_dir, f"{self.arch}")
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        # Save the aggregated outputs:
        np.savez(
            os.path.join(out_dir, f"conf_{self.target_dataset.lower()}.npz"),
            labels=labels.detach().cpu().numpy(),
            outputs=outputs.detach().cpu().numpy(),
            indices=idx.detach().cpu().numpy(),
            features=features.detach().cpu().numpy()
        )

        # Optionally, clear the stored outputs:
        self.test_outputs.clear()


    def configure_optimizers(self):
        pass