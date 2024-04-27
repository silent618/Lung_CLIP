import torch
import torch.nn as nn


# 定义一个简单的两层MLP模型
class MLP(nn.Module):
    def __init__(self, hparams):
        super(MLP, self).__init__()
        input_dim = hparams.input_dim  # 后续需要调整
        hidden_dim = hparams.hidden_dim
        output_dim = hparams.output_dim
        # 输入层到隐藏层的线性变换
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        # 隐藏层到输出层的线性变换
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 前向传播
        x = self.relu(self.layer1(x))  # 隐藏层的线性变换加上ReLU激活
        x = self.layer2(x)  # 输出层的线性变换
        return x
