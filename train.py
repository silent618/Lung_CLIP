import torch
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
    # 参数设置
    hparams = {
        "model": "resnet50",  # 从timm库选择的模型
        "embedding_dim": 128,
        "projection_dim": 128,
        "temperature": 0.1,
        "batch_size": 32,
    }

    # 初始化对比学习框架
    framework = SimCLR(hparams)

    # 数据变换示例，可根据需要调整
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 实例化数据集
    dataset = ImageTableDataset(image_paths, table_data, transform=transform)
    data_loader = DataLoader(dataset, batch_size=hparams["batch_size"], shuffle=True)

    # 设置优化器等
    framework.optimizer = torch.optim.Adam(list(framework.encoder_imaging.parameters())
                                           + list(framework.projector_imaging.parameters())
                                           + list(framework.table_encoder.parameters())
                                           + list(framework.table_projector.parameters()), lr=0.001)

    # 训练循环
    resume_training = True
    if resume_training:
        framework, framework.optimizer, start_epoch, last_loss = load_checkpoint(framework, framework.optimizer)
    else:
        start_epoch = 0
        last_loss = None

    num_epochs = 20  # 假定总共训练20个epoch，根据实际情况调整

    # 训练循环
    for epoch in range(start_epoch, num_epochs):
        loss = framework.train_step(data_loader)
        print(f"Epoch {epoch}, Loss: {loss}")

        # 每个epoch结束后保存检查点
        save_checkpoint(framework, framework.optimizer, epoch, loss)

    # 初始化下游任务模型
    downstream_model = DownstreamClassifier()

    # 微调
    optimizer = torch.optim.SGD(downstream_model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()  # 分类任务的标准损失函数

    for epoch in range(10):
        for img_data, table_data in data_loader:
            img_features = framework.encoder_imaging(img_data)
            table_features = table_data[:-1]
            labels = table_data[-1]

            outputs = downstream_model(img_features, table_features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 评估
    correct = 0
    total = 0
    with torch.no_grad():
        for img_data, table_data in test_dataloader:
            img_features = framework.encoder_imaging(img_data)
            table_features = table_data[:-1]
            labels = table_data[-1]
            outputs = downstream_model(img_features, table_features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the model on the {total} test images: {accuracy}%")

