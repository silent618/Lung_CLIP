import torch
import torch.nn.functional as F


class NT_XentLoss(torch.nn.Module):
    def __init__(self, batch_size, temperature=0.5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(NT_XentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, z_i, z_j):
        # 计算两组嵌入间的点积相似度矩阵
        sim_matrix = torch.matmul(z_i, z_j.T) / self.temperature

        # 对角线上的元素是自身与自身的相似度，应设置为非常小的负数，以避免自我对比
        mask = torch.eye(self.batch_size, device=self.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, value=float('-inf'))

        # 计算softmax以获得每个样本与其他样本的相似度概率分布
        sim_probs = self.softmax(sim_matrix)

        # 只取上三角或下三角的部分，因为矩阵是对称的，且已排除对角线
        sim_probs = sim_probs.triu(diagonal=1)

        # 计算每行的损失（除了自己以外的所有负对的log似然）
        loss_per_row = -torch.log(sim_probs.sum(dim=1))

        # 平均所有样本的损失
        loss = loss_per_row.mean()
        return loss