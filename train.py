import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from timm.models import create_model
from lightly.models.modules import SimCLRProjectionHead
from TabularEncoder import MLP
from dataset import ImageTableDataset
from Loss import NT_XentLoss
from SimCLR import SimCLR
from Classifier import DownstreamClassifier
from utils import load_checkpoint, save_checkpoint

if __name__ == "__main__":
    # Read Hyper Params
    with open('config.yaml', 'r', encoding='utf-8') as file:
        hparams = yaml.safe_load(file)

    # 初始化对比学习框架
    framework = SimCLR(hparams)

    # 数据变换示例，可根据需要调整
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.35),
        transforms.RandomVerticalFlip(p=0.35),
        transforms.RandomErasing(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 实例化数据集
    dataset = ImageTableDataset(hparams.image_path, hparams.table_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=hparams.batch_size, shuffle=True)

    # 设置优化器等
    framework.optimizer = torch.optim.Adam(list(framework.encoder_imaging.parameters())
                                           + list(framework.projector_imaging.parameters())
                                           + list(framework.encoder_tabular.parameters())
                                           + list(framework.projector_tabular.parameters()), hparams.lr)

    # 训练循环
    if hparams.resume_training:
        framework, framework.optimizer, start_epoch, last_loss = load_checkpoint(framework,
                                                                                 framework.optimizer,
                                                                                 hparams)
    else:
        start_epoch = 0
        last_loss = None

    # 训练循环
    for epoch in range(start_epoch, hparams.num_epochs):
        loss = framework.train_step(data_loader)
        print(f"Epoch {epoch}, Loss: {loss}")

        # 每10个epoch结束后保存检查点
        if (epoch + 1) % 10 == 0:
            save_checkpoint(framework, framework.optimizer, epoch, loss)

    # 初始化下游任务模型
    downstream_model = DownstreamClassifier(hparams.Classifier)

    # 微调
    optimizer = torch.optim.SGD(downstream_model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()  # 分类任务的标准损失函数

    for epoch in range(hparams.Classifier.epochs):
        for img_data, table_data in data_loader:
            img_features = framework.encoder_imaging(img_data)
            table_features = table_data[:-3]
            t_stage_labels = table_data[-3]
            n_stage_labels = table_data[-2]
            m_stage_labels = table_data[-1]

            t_stage, n_stage, m_stage = downstream_model(img_features, table_features)
            loss1 = criterion(t_stage, t_stage_labels)
            loss2 = criterion(n_stage, n_stage_labels)
            loss3 = criterion(m_stage, m_stage_labels)
            loss = loss1 + loss2 + loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 评估
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = ImageTableDataset(hparams.test_image_path, hparams.test_table_path, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=hparams.test_batch_size, shuffle=False)
    t_stage_correct = 0
    n_stage_correct = 0
    m_stage_correct = 0
    all_stage_correct = 0
    total = 0
    with torch.no_grad():
        for img_data, table_data in test_dataloader:
            img_data.cuda()
            table_data.cuda()
            img_features = framework.encoder_imaging(img_data)
            table_features = table_data[:-3]
            t_stage_labels = table_data[-3]
            n_stage_labels = table_data[-2]
            m_stage_labels = table_data[-1]
            t_stage, n_stage, m_stage = downstream_model(img_features, table_features)
            _, t_predicted = torch.max(t_stage.data, 1)
            _, n_predicted = torch.max(n_stage.data, 1)
            _, m_predicted = torch.max(m_stage.data, 1)
            total += t_stage_labels.size(0)
            t_stage_correct += (t_predicted == t_stage_labels).sum().item().cpu()
            n_stage_correct += (n_predicted == n_stage_labels).sum().item().cpu()
            m_stage_correct += (m_predicted == m_stage_labels).sum().item().cpu()
            all_stage_correct += (t_predicted == t_stage_labels
                                  and n_predicted == n_stage_labels
                                  and m_predicted == m_stage_labels).sum().item().cpu()

    t_stage_accuracy = 100 * t_stage_correct / total
    n_stage_accuracy = 100 * n_stage_correct / total
    m_stage_accuracy = 100 * m_stage_correct / total
    all_stage_accuracy = 100 * all_stage_correct / total
    print(f"T-stage accuracy of the model on the {total} test images: {t_stage_accuracy}%")
    print(f"N-stage accuracy of the model on the {total} test images: {n_stage_accuracy}%")
    print(f"M-stage accuracy of the model on the {total} test images: {m_stage_accuracy}%")
    print(f"All-stage accuracy of the model on the {total} test images: {all_stage_accuracy}%")
