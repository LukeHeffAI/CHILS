import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, List
import logging
import clip

from src.data_utils import get_dataset

log = logging.getLogger("app")

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        target_dataset: str = "Imagenet",
        batch_size: int = 128,
        num_classes: int = 1000,
        clip_transform: str = 'ClipViTL14'
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        # we'll wrap the single string into a list so we can support multiple tests if desired
        self.target_dataset = [target_dataset]

        # prepare CLIP preprocessing transforms
        self.train_transform = None
        self.val_transform = None
        self.test_transform = []

        # load the appropriate CLIP preprocess for all splits
        if clip_transform == 'ClipViTL14':
            _, preprocess = clip.load("ViT-L/14@336px",  device='cuda')
        elif clip_transform == 'ClipViTB32':
            _, preprocess = clip.load("ViT-B/32",        device='cuda')
        elif clip_transform == 'ClipViTB16':
            _, preprocess = clip.load("ViT-B/16",        device='cuda')
        elif clip_transform == 'ClipRN50x4':
            _, preprocess = clip.load("RN50x4",          device='cuda')
        elif clip_transform == 'ClipRN101':
            _, preprocess = clip.load("RN101",           device='cuda')
        elif clip_transform == 'ClipRN50':
            _, preprocess = clip.load("RN50",            device='cuda')
        else:
            raise NotImplementedError(f"invalid CLIP model: {clip_transform}")

        # use the same transform for everything by default
        self.train_transform = preprocess
        self.val_transform   = preprocess
        self.test_transform  = [preprocess for _ in self.target_dataset]

    def setup(self, stage: Optional[str] = None):
        # called by Lightning at the start of fit/test
        if stage in (None, 'fit'):
            log.info("Creating train/val datasets …")
            # for now we only train/validate on the *first* target_dataset
            ds = self.target_dataset[0]
            self.train_dataset = get_dataset(
                data_dir=self.data_dir,
                dataset=ds,
                train=True,
                transform=self.train_transform
            )
            self.val_dataset = get_dataset(
                data_dir=self.data_dir,
                dataset=ds,
                train=False,
                transform=self.val_transform
            )
            log.info(f" → train: {len(self.train_dataset)}  val: {len(self.val_dataset)}")

        if stage in (None, 'test'):
            log.info("Creating test datasets …")
            self.test_dataset = []
            for i, ds in enumerate(self.target_dataset):
                test_ds = get_dataset(
                    data_dir=self.data_dir,
                    dataset=ds,
                    train=False,
                    transform=self.test_transform[i]
                )
                self.test_dataset.append(test_ds)
                log.info(f" → test[{i}] '{ds}': {len(test_ds)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def test_dataloader(self):
        # returns a list of loaders, one per target_dataset
        return [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            for ds in self.test_dataset
        ]