import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImageTableDataset

if __name__ == "__main__":
    # Read Hyper Params
    with open('config.yaml', 'r', encoding='utf-8') as file:
        hparams = yaml.safe_load(file)

    framework = torch.load(hparams.load_path)
    classifier = torch.load(hparams.Classifier.load_path)

    # 数据变换示例，可根据需要调整
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
    cancer_type_correct = 0
    all_stage_correct = 0
    total = 0
    with torch.no_grad():
        for img_data1, img_data2, table_data in test_dataloader:
            img_data1.cuda()
            img_data2.cuda()
            table_data.cuda()
            img_features1 = framework.encoder_imaging(img_data1)
            img_features2 = framework.encoder_imaging(img_data2)
            img_features = framework.cross_attention(img_features1, img_features2)
            table_features = table_data[:-5]
            t_stage_labels = table_data[-4]
            n_stage_labels = table_data[-3]
            m_stage_labels = table_data[-2]
            cancer_type_labels = table_data[-1]
            t_stage, n_stage, m_stage, cancer_type = classifier(img_features, table_features)
            _, t_predicted = torch.max(t_stage.data, 1)
            _, n_predicted = torch.max(n_stage.data, 1)
            _, m_predicted = torch.max(m_stage.data, 1)
            _, cancer_type_predicted = torch.max(cancer_type.data, 1)
            total += t_stage_labels.size(0)
            t_stage_correct += (t_predicted == t_stage_labels).sum().item()
            n_stage_correct += (n_predicted == n_stage_labels).sum().item()
            m_stage_correct += (m_predicted == m_stage_labels).sum().item()
            cancer_type_correct += (cancer_type_predicted == cancer_type_labels).sum().item()
            all_stage_correct += (t_predicted == t_stage_labels
                                  and n_predicted == n_stage_labels
                                  and m_predicted == m_stage_labels
                                  and cancer_type_predicted == cancer_type_labels).sum().item()

        t_stage_accuracy = 100 * t_stage_correct / total
        n_stage_accuracy = 100 * n_stage_correct / total
        m_stage_accuracy = 100 * m_stage_correct / total
        cancer_type_accuracy = 100 * cancer_type_correct / total
        all_stage_accuracy = 100 * all_stage_correct / total
        print(f"T-stage accuracy of the model on the {total} test images: {t_stage_accuracy}%")
        print(f"N-stage accuracy of the model on the {total} test images: {n_stage_accuracy}%")
        print(f"M-stage accuracy of the model on the {total} test images: {m_stage_accuracy}%")
        print(f"Cancer type accuracy of the model on the {total} test images: {cancer_type_accuracy}%")
        print(f"All-stage accuracy of the model on the {total} test images: {all_stage_accuracy}%")
