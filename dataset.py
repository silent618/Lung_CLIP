import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os


def get_subfolders_listdir(path):
    all_entries = os.listdir(path)
    subfolders = [entry for entry in all_entries if os.path.isdir(entry)]
    return subfolders


class ImageTableDataset(Dataset):
    def __init__(self, image_path, table_path, transform=None):
        """
        Args:
            image_path : 图像文件夹路径。
            table_path : 表格数据路径.
            transform (callable, optional): 对图像进行变换的可选函数。
        """
        image_folders = get_subfolders_listdir(image_path)
        image_folders.sort(key=lambda x: int(x.split('.')[0]))
        self.image_folders = image_folders
        self.table_data = pd.read_csv(table_path, encoding='utf-8')
        self.transform = transform

    def __len__(self):
        return len(self.image_folders)

    def __getitem__(self, idx):
        # 图像读取和预处理
        img_folder = self.image_folders[idx]
        img_paths = os.listdir(img_folder)
        img1 = Image.open(img_paths[0])
        img2 = Image.open(img_paths[1])
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # 表格数据获取
        table_sample = self.table_data.iloc[idx] if isinstance(self.table_data, pd.DataFrame) else self.table_data[idx]
        table_tensor = torch.tensor(table_sample.values, dtype=torch.float32)

        return img1, img2, table_tensor
