import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from timm.models import create_model
from lightly.models.modules import SimCLRProjectionHead
from TabularEncoder import MLP
from dataset import ImageTableDataset
from Loss import NT_XentLoss


class SimCLR:
    def __init__(self, hparams, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.hparams = hparams
        self.device = device
        self.initialize_imaging_encoder_and_projector()
        # 假设还有表格数据的初始化函数
        self.initialize_table_encoder_and_projector()
        self.criterion = NT_XentLoss(self.hparams.batch_size, self.hparams.temperature)  # 假设实现了NT-XentLoss

    def initialize_imaging_encoder_and_projector(self) -> None:
        self.encoder_imaging = create_model(self.hparams.model, pretrained=True, features_only=True)
        self.pooled_dim = self.encoder_imaging.feature_info[-1]['num_chs']
        self.projector_imaging = SimCLRProjectionHead(self.pooled_dim, self.hparams.embedding_dim,
                                                      self.hparams.projection_dim)

    def initialize_table_encoder_and_projector(self):
        self.encoder_tabular = MLP(self.hparams)
        self.projector_tabular = SimCLRProjectionHead(self.hparams.embedding_dim, self.hparams.embedding_dim,
                                                      self.hparams.projection_dim)

    def forward(self, img_batch, table_batch):
        img_features = self.encoder_imaging(img_batch)
        img_proj = self.projector_imaging(img_features)

        table_features = self.encoder_tabular(table_batch)
        table_proj = self.projector_tabular(table_features)

        return img_proj, table_proj

    def train_step(self, data_loader):
        running_loss = 0.0
        for img_batch, table_batch in data_loader:
            img_batch.cuda()
            table_batch.cuda()
            img_proj, table_proj = self.forward(img_batch, table_batch)
            loss = self.criterion(img_proj, table_proj)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        return avg_loss

    def state_dict(self):
        """
        返回包含模型状态的字典，包括图像和表格编码器及投影头的参数。
        """
        state = {
            'encoder_imaging_state_dict': self.encoder_imaging.state_dict(),
            'projector_imaging_state_dict': self.projector_imaging.state_dict(),
            'encoder_tabular_state_dict': self.encoder_tabular.state_dict(),
            'projector_tabular_state_dict': self.projector_tabular.state_dict()
        }
        return state
