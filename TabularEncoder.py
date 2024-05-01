import torch
import torch.nn as nn


# 定义一个简单的两层MLP模型
class MLP(nn.Module):
    def __init__(self, hparams, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(MLP, self).__init__()
        self.device = device
        input_dim = hparams.input_dim  # 后续需要调整
        hidden_dim1 = hparams.hidden_dim1
        hidden_dim2 = hparams.hidden_dim2
        output_dim = hparams.embedding_dim
        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim1),
                                nn.ReLU(),
                                nn.Linear(hidden_dim1, hidden_dim2),
                                nn.ReLU(),
                                nn.Linear(hidden_dim2, output_dim),
                                nn.ReLU())

    def forward(self, x):
        output = self.fc(x)
        return output
