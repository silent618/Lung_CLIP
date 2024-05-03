import torch
from timm.models import create_model
from lightly.models.modules import SimCLRProjectionHead
from TabularEncoder import MLP
from Loss import NT_XentLoss

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionLayer(nn.Module):
    def __init__(self, dim=1024):
        super(CrossAttentionLayer, self).__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

    def forward(self, feat_v, feat_s):
        Q = self.query_proj(feat_v.flatten(2)).transpose(1, 2)  # (batch_size, height*width, channels)
        K = self.key_proj(feat_s.flatten(2)).transpose(1, 2)  # 同上
        V = self.value_proj(feat_s.flatten(2)).transpose(1, 2)  # 同上

        # 计算注意力权重
        attn_weights = torch.bmm(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(K.shape[-1]).float())
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 使用注意力权重融合特征
        fused_feat = torch.bmm(attn_weights, V)

        # 将融合后的特征重新塑形回原始维度
        fused_feat = fused_feat.transpose(1, 2).reshape(feat_v.shape)

        return fused_feat


class SimCLR:
    def __init__(self, hparams, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.hparams = hparams
        self.device = device

        self.initialize_imaging_encoder_and_projector()
        self.initialize_table_encoder_and_projector()
        self.criterion = NT_XentLoss(self.hparams.batch_size, self.hparams.temperature)  # 假设实现了NT-XentLoss

    def initialize_imaging_encoder_and_projector(self) -> None:
        self.encoder_imaging1 = create_model(self.hparams.model, pretrained=True, features_only=True)
        self.encoder_imaging2 = create_model(self.hparams.model, pretrained=True, features_only=True)
        self.cross_attention = CrossAttentionLayer()
        self.pooled_dim = self.encoder_imaging1.feature_info[-1]['num_chs']
        self.projector_imaging = SimCLRProjectionHead(self.pooled_dim, self.hparams.embedding_dim,
                                                      self.hparams.projection_dim)

    def initialize_table_encoder_and_projector(self):
        self.encoder_tabular = MLP(self.hparams)
        self.projector_tabular = SimCLRProjectionHead(self.hparams.embedding_dim, self.hparams.embedding_dim,
                                                      self.hparams.projection_dim)

    def forward(self, img_batch1, img_batch2, table_batch):
        img_features1 = self.encoder_imaging1(img_batch1)
        img_features2 = self.encoder_imaging2(img_batch2)
        img_features = self.cross_attention(img_features1, img_features2)
        img_proj = self.projector_imaging(img_features)

        table_features = self.encoder_tabular(table_batch)
        table_proj = self.projector_tabular(table_features)

        return img_proj, table_proj

    def train_step(self, data_loader):
        running_loss = 0.0
        for img_batch1, img_batch2, table_batch in data_loader:
            img_batch1.cuda()
            img_batch2.cuda()
            table_batch.cuda()
            img_proj, table_proj = self.forward(img_batch1, img_batch2, table_batch)
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
            'encoder_imaging_state_dict1': self.encoder_imaging1.state_dict(),
            'encoder_imaging_state_dict2': self.encoder_imaging2.state_state(),
            'cross_attention_state_dict': self.cross_attention.state_dict(),
            'projector_imaging_state_dict': self.projector_imaging.state_dict(),
            'encoder_tabular_state_dict': self.encoder_tabular.state_dict(),
            'projector_tabular_state_dict': self.projector_tabular.state_dict()
        }
        return state
