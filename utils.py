import torch


def save_checkpoint(model, optimizer, epoch, loss, hparams):
    """保存模型状态到检查点文件"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, hparams.save_path)


def load_checkpoint(model, optimizer, hparams):
    """从检查点文件恢复模型状态"""
    checkpoint = torch.load(hparams.load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss
