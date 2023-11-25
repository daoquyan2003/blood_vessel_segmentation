from typing import Optional, Tuple

import albumentations as A
import torch
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from src.data.blood_vessel.components.sennet_hoa_dataset import SenNetHOADataset
from src.data.blood_vessel.components.transform_sennet_hoa import TransformSenNetHOA

class SenNetHOADataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = SenNetHOADataset(data_dir=self.hparams.data_dir)
            data_len = len(dataset)
            train_len = int(data_len * self.hparams.train_val_test_split[0])
            val_len = int(data_len * self.hparams.train_val_test_split[1])
            test_len = data_len - train_len - val_len

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=[train_len, val_len, test_len],
                generator=torch.Generator().manual_seed(42),
            )

            self.data_train = TransformSenNetHOA(self.data_train)
            self.data_val = TransformSenNetHOA(self.data_val)
            self.data_test = TransformSenNetHOA(self.data_test)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )