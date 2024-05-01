import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os


class ImageTableDataset(Dataset):
    def __init__(self, image_path, table_path, transform=None):
        """
        Args:
            image_path : 图像文件路径。
            table_path : 表格数据路径.
            transform (callable, optional): 对图像进行变换的可选函数。
        """
        image_paths = os.listdir(image_path)
        image_paths.sort(key=lambda x: int(x.split('.')[0]))
        self.image_paths = image_paths
        self.table_data = pd.read_csv(table_path, encoding='utf-8')
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 图像读取和预处理
        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)

        # 表格数据获取
        table_sample = self.table_data.iloc[idx] if isinstance(self.table_data, pd.DataFrame) else self.table_data[idx]
        table_tensor = torch.tensor(table_sample.values, dtype=torch.float32)

        return img, table_tensor
