import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class ImageTableDataset(Dataset):
    def __init__(self, image_paths, table_data, transform=None):
        """
        Args:
            image_paths (list): 图像文件路径列表。
            table_data (list or pd.DataFrame): 表格数据列表或DataFrame。
            transform (callable, optional): 对图像进行变换的可选函数。
        """
        self.image_paths = image_paths
        self.table_data = table_data
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
