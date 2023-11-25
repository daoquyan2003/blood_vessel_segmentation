from typing import Any, Optional

import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from src.data.blood_vessel.components.sennet_hoa_dataset import SenNetHOADataset

class TransformSenNetHOA(Dataset):
    mean = None
    std = None

    def __init__(self, dataset: SenNetHOADataset, transform: Optional[Compose] = None) -> None:
        super().__init__()

        self.dataset = dataset

        if transform is not None:
            self.transform = transform
        else:
            self.transform = Compose(
                [
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, p=0.5),
                    A.Affine(scale=0.9, translate_percent=0.9, rotate=(-15, 15), p=0.3),
                    A.Flip(p=0.3),
                    A.Resize(576, 576),
                    A.Normalize(mean=[0.5], std=[0.5]),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Any:
        image, mask = self.dataset[index]
        image = np.array(Image.fromarray(image))
        mask = np.array(Image.fromarray(mask))

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            # img_size set in hydra config
            image = transformed["image"]  # (3, img_size, img_size)
            mask = transformed["mask"]  # (img_size, img_size), uint8
            # mask = mask.unsqueeze(0).float()  # (1, img_size, img_size)
            mask = mask.permute(2, 0, 1)

        return image, mask

if __name__ == "__main__":
    dataset = SenNetHOADataset(data_dir="data")
    transformed_dataset = TransformSenNetHOA(dataset=dataset)
    img, mask = transformed_dataset[2]
    print(img.shape)
    print(mask.shape)
