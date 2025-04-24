import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix, MeanMetric
import torch
import torch.optim.lr_scheduler as lr_sched
from torch.nn.functional import softmax, one_hot

from typing import List, Optional
from src.model_utils import *
import logging 
import wandb
import numpy as np

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
        hash: Optional[str] = None
    ):
        super().__init__()

        self.arch = arch
        self.model = get_model(arch = arch, dataset = base_task , pretrained= pretrained, retrain=False, extract_features=True, work_dir=work_dir)
        
        if classnames is None:
            raise ValueError("You must pass dataset classnames into EvalNet")
        self.model.init_text(base_task.lower(), classnames)
        self.target_dataset = target_dataset

        # self.pred_acc = nn.ModuleList([Accuracy() for _ in self.target_dataset])

        # self.confusion_matrix = ConfusionMatrix(1000)
        
        self.work_dir = work_dir
        self.hash = hash
        self.pretrained = pretrained

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, _ = batch[:2]
        return self.model(x)

    def process_batch(self, batch, stage="train", dataloader_idx=0):
        x, y, idx = batch[:3]

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