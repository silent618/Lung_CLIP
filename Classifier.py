import torch
from torch import nn


class DownstreamClassifier(nn.Module):
    def __init__(self, classifier_hparams, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(DownstreamClassifier, self).__init__()
        self.device = device
        self.image_fc = nn.Sequential(nn.Linear(classifier_hparams.image_feature_dim, classifier_hparams.hid_dim1),
                                      nn.ReLU(),
                                      nn.Linear(classifier_hparams.hid_dim1, classifier_hparams.hid_dim2),
                                      nn.ReLU())  # 图像特征到分类的全连接层
        self.table_fc = nn.Sequential(nn.Linear(classifier_hparams.table_feature_dim, classifier_hparams.hid_dim1),
                                      nn.ReLU(),
                                      nn.Linear(classifier_hparams.hid_dim1, classifier_hparams.hid_dim2),
                                      nn.ReLU())  # 表格特征到分类的全连接层
        self.task_T_stage = nn.Linear(2*classifier_hparams.hid_dim2, classifier_hparams.T_stage_num_classes)
        self.task_N_stage = nn.Linear(2 * classifier_hparams.hid_dim2, classifier_hparams.N_stage_num_classes)
        self.task_M_stage = nn.Linear(2 * classifier_hparams.hid_dim2, classifier_hparams.M_stage_num_classes)
        self.task_cancer_type = nn.Linear(2 * classifier_hparams.hid_dim2, classifier_hparams.cancer_type_num_classes)
        self.softmax = torch.nn.Softmax()

    def forward(self, img_features, table_features):
        img_output = self.image_fc(img_features)
        table_output = self.table_fc(table_features)
        # 整合两种特征
        combined_feature = torch.cat(img_output, table_output)
        t_stage = self.softmax(self.task_T_stage(combined_feature))
        n_stage = self.softmax(self.task_N_stage(combined_feature))
        m_stage = self.softmax(self.task_M_stage(combined_feature))
        cancer_type = self.softmax(self.task_cancer_type(combined_feature))
        return t_stage, n_stage, m_stage, cancer_type
