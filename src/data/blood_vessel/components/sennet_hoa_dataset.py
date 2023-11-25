import numpy as np
import pandas as pd 
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os
from src.utils.sennet_hoa.sennet_hoa_utils import rle_decode

class SenNetHOADataset(Dataset):
    def __init__(self, data_dir) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.image_dir = os.path.join(self.data_dir, "train")
        masks = pd.read_csv(os.path.join(self.data_dir, "train_rles.csv"))
        masks[["dataset", "slice"]] = masks['id'].str.rsplit(pat='_', n=1, expand=True)
        
        self.img_paths = []
        self.rles = []

        corrupted_file = 0

        for _, row in masks.iterrows():
            p_img = os.path.join(self.image_dir, row["dataset"], "images", f'{row["slice"]}.tif')
            if not os.path.isfile(p_img):
                corrupted_file += 1
                continue
            self.img_paths.append(p_img)
            self.rles.append(row['rle'])

    def __len__(self) -> int:
        assert len(self.img_paths) == len(self.rles)
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple:
        img = plt.imread(self.img_paths[idx])
        if img.ndim == 3:
            img = np.mean(img, axis=2)
        if np.max(img) > 255:
            img = np.clip(img / 255, 0, 255).astype(np.uint8)
        mask = rle_decode(self.rles[idx], img_shape=img.shape)
        # mask = np.transpose(mask, (2, 0, 1))

        return img, mask

if __name__ == "__main__":
    dataset = SenNetHOADataset(data_dir="data")
    img, mask = dataset[2]
    print(img.shape)
    print(mask.shape)
        


