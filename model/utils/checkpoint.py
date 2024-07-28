import torch
import torch.nn as nn


def load_model_checkpoint(model: nn.Module,
                          checkpoint_path: str,
                          strict: bool = True):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    return checkpoint


def load_optimizer_checkpoint(optimizer: torch.optim.Optimizer,
                              checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


def save_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    loss: float,
                    checkpoint_path: str):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint
