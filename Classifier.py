import torch
from torch import nn


class DownstreamClassifier(nn.Module):
    def __init__(self):
        # 后续调整
        super(DownstreamClassifier, self).__init__()
        self.image_fc = nn.Linear(image_embedding_size, hid_size)  # 图像特征到分类的全连接层
        self.table_fc = nn.Linear(table_embedding_size, hid_size)  # 表格特征到分类的全连接层
        self.all_fc = nn.Linear(2*hid_size, num_classes)
        self.softmax = torch.nn.Softmax()

    def forward(self, img_features, table_features):
        img_output = self.image_fc(img_features)
        table_output = self.table_fc(table_features)
        # 可以是简单平均、加权融合或者其他更复杂的方式整合两种特征
        combined_feature = torch.cat(img_output, table_output)
        combined_output = self.all_fc(combined_feature)
        combined_output = self.softmax(combined_output)
        return combined_output
